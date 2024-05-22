from datetime import datetime
from typing import Any, Dict

import streamlit as st
from dotenv import find_dotenv, load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.callbacks import get_openai_callback
from langchain.callbacks.base import BaseCallbackHandler
from langchain.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI


class SearchCallbackHandler(BaseCallbackHandler):
    tool_outputs: list[tuple[str, Any]] = []

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> Any:
        """Run when tool starts running."""
        name = serialized["name"]
        if name == "current_datetime":
            st.write("Getting current time...")
        elif name == "duckduckgo_search":
            inputs = kwargs["inputs"]
            st.write(f"Searching for '{inputs['query']}'")
        
        return super().on_tool_start(serialized, input_str, **kwargs)

    def on_tool_end(self, output: Any, **kwargs: Any) -> Any:
        """Run when tool ends running."""
        self.tool_outputs.append((kwargs["name"], output))
        return super().on_tool_end(output, **kwargs)


@tool
def current_datetime() -> str:
    """Returns the current datetime."""
    return datetime.now().isoformat()


# Load environment variables
load_dotenv(find_dotenv())

# Set up LangChain components
llm = ChatOpenAI(model="gpt-4o", temperature=0)
tools = [
    # DuckDuckGoSearchRun(),
    TavilySearchResults(max_results=3),
    current_datetime
]

base_prompt = hub.pull("langchain-ai/openai-functions-template")
instructions = """
You are a helpful assistant. Your task is to understand the user's question and provide accurate, factual answers. If the question requires worldwide or updated data, use the search and date-time functions to obtain the necessary information. For questions that require comparisons, query data for each statement as needed.

Follow these steps to answer a question:
1. Read the question.
2. Thoroughly analyze the question.
3. Break down long questions into sub-questions.
4. Query data for sub-questions if your existing knowledge is insufficient.
5. Provide a detailed answer based on the data.

Query Data Note:
- If you are going to query a data, expand and elaborate the query and then pass it to the search action.
- For time-related data, use the 'current_datetime' function to obtain the current date and time, and include this information in your query.
- If you cannot find the needed data or sufficient information, mention a reference or source where the user can find the data themselves.
"""
prompt = base_prompt.partial(instructions=instructions)

agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, stream_runnable=False, verbose=True)

# Setup Streamlit
st.set_page_config(page_title="LLM With WebSearch")
st.title("LLM With WebSearch")
st.info("`I am an AI that can answer questions by exploring, reading, and summarizing web pages.`")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.chat_history = []


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

chat_container = st.empty()

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Write answer and sources
    callback = SearchCallbackHandler()

    with st.status("Generating response...", expanded=True) as status:
        with get_openai_callback() as cb:
            result = agent_executor.invoke({"input": prompt, "chat_history": st.session_state.chat_history}, config={"callbacks": [callback]})
            print(f"Total Tokens: {cb.total_tokens}")
            print(f"Prompt Tokens: {cb.prompt_tokens}")
            print(f"Completion Tokens: {cb.completion_tokens}")
            total_cost = (cb.prompt_tokens*0.005 + cb.completion_tokens*0.015) / 1000
            print(f"Total Cost (USD): ${total_cost:.6f}")

            status.update(label="Response generated.", state="complete", expanded=False)

    response_content = result["output"]
    sources = []
    tool_outputs_msg = ""
    for tool_name, tool_output in callback.tool_outputs:
        if tool_name == "tavily_search_results_json":
            sources.extend(tool_output)
        tool_outputs_msg += f"{tool_name}: {tool_output}\n"

    st.session_state.chat_history.extend(
        [
            HumanMessage(content=result["input"]),
            AIMessage(content=tool_outputs_msg),
            AIMessage(content=result["output"]),
        ]
    )

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response_content)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response_content})
