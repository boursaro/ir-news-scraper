import streamlit as st
from dotenv import find_dotenv, load_dotenv
from langchain.callbacks.base import BaseCallbackHandler
from langchain_community.retrievers import TavilySearchAPIRetriever
from langchain.callbacks.tracers import ConsoleCallbackHandler
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI


load_dotenv(find_dotenv())

st.set_page_config(page_title="Interweb Explorer", page_icon="🌐")


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.info(self.text)


st.header("`Interweb Explorer`")
st.info("`I am an AI that can answer questions by exploring, reading, and summarizing web pages."
    "I can be configured to use different modes: public API or private (no data sharing).`")

# Make retriever and llm
retriever = TavilySearchAPIRetriever(k=4)

prompt = ChatPromptTemplate.from_template(
"""Answer the question based only on the context provided.

Context: {context}

Question: {question}"""
)

qa_chain = (
    RunnablePassthrough.assign(context=(lambda x: x["question"]) | retriever)
    | prompt
    | ChatOpenAI(model="gpt-4-turbo")
    | StrOutputParser()
)

# User input 
question = st.text_input("`Ask a question:`")

if question:
    # Generate answer (w/ citations)
    import logging
    logging.basicConfig()
    logging.getLogger("langchain.retrievers.web_research").setLevel(logging.INFO)    

    # Write answer and sources
    answer = st.empty()
    stream_handler = StreamHandler(answer, initial_text="`Answer:`\n\n")
    result = qa_chain.invoke({"question": question},config={"callbacks":[stream_handler, ConsoleCallbackHandler()]})
    answer.info('`Answer:`\n\n' + result)
