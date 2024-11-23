import os
from typing import TypedDict, Annotated

from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from tavily import TavilyClient

from openai import OpenAI
from langgraph.graph import add_messages, StateGraph

load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")
tavily = os.getenv("TAVILY_API_KEY")

llm_name = "gpt-3.5-turbo"

# client = OpenAI(api_key=openai_key)
# model = ChatOpenAI(api_key=openai_key, model=llm_name)

MODEL_NAME = 'llama-3.1-8b-instant'
model = ChatGroq(model_name=MODEL_NAME, api_key=os.getenv('GROQ_API_KEY'))


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)

# Create tools
search_tool = TavilySearchResults(max_results=2)

res = search_tool.invoke("What is the capital of the UK?")
print(res)

print(">>>>>>>>><<<<<<<<<<<")

tavily_search = TavilyClient(api_key=tavily)

response = tavily_search.search(query="What is the unit of central tendency?", max_results=2)
print(response)



