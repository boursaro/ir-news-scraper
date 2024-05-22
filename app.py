import json
from typing import Any, List
from uuid import UUID

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from datetime import datetime
from dotenv import find_dotenv, load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.callbacks import BaseCallbackHandler
from langchain.callbacks import get_openai_callback
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain.tools import BaseTool

chat_history = []


class CurrentDatetimeTool(BaseTool):
    def __init__(self):
        super().__init__(name="current_datetime", description="Returns the current datetime")

    def _run(self, *args, **kwargs):
        return {"datetime": datetime.now().isoformat()}

    def _arun(self, *args, **kwargs):
        raise NotImplementedError("This tool does not support async")


class SearchCallbackHandler(BaseCallbackHandler):
    tool_outputs: list[tuple[str, Any]] = []

    def on_tool_end(self, output: Any, **kwargs: Any) -> Any:
        """Run when tool ends running."""
        self.tool_outputs.append((kwargs["name"], output))
        return super().on_tool_end(output, **kwargs)


# Load environment variables
load_dotenv(find_dotenv())

# Initialize Flask app
app = Flask(__name__, template_folder='')
CORS(app)

# Set up LangChain components
llm = ChatOpenAI(model="gpt-4o", temperature=0)
tools = [
    DuckDuckGoSearchRun(),
    TavilySearchResults(max_results=3),
    CurrentDatetimeTool()
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

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/api/message/', methods=['POST'])
def handle_message():
    global chat_history

    data = request.json
    user_message = data.get('message', '')

    if not user_message:
        return jsonify({"error": "Message content is required"}), 400

    # Generate the assistant response
    # try:
    callback = SearchCallbackHandler()
    with get_openai_callback() as cb:
        result = agent_executor.invoke({"input": user_message, "chat_history": chat_history}, config={"callbacks": [callback]})
        print(f"Total Tokens: {cb.total_tokens}")
        print(f"Prompt Tokens: {cb.prompt_tokens}")
        print(f"Completion Tokens: {cb.completion_tokens}")
        total_cost = (cb.prompt_tokens*0.005 + cb.completion_tokens*0.015) / 1000
        print(f"Total Cost (USD): ${total_cost:.6f}")

    response_content = result["output"]
    sources = []
    tool_outputs_msg = ""
    for tool_name, tool_output in callback.tool_outputs:
        if tool_name == "tavily_search_results_json":
            sources.extend(tool_output)
        tool_outputs_msg += f"{tool_name}: {tool_output}\n"

    chat_history.extend(
        [
            HumanMessage(content=result["input"]),
            AIMessage(content=tool_outputs_msg),
            AIMessage(content=result["output"]),
        ]
    )

    # except Exception as e:
    #     return jsonify({"error": str(e)}), 500

    # Prepare the response data
    response_data = {
        "response": response_content,
        "sources": sources,
    }

    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True, port=8080)
