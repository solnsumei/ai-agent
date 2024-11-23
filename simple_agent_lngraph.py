import os
from pprint import pprint
from typing import TypedDict, Annotated

from dotenv import load_dotenv
from langchain_community.tools import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

from openai import OpenAI
from langgraph.graph import add_messages, StateGraph

load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")
tavily = os.getenv("TAVILY_API_KEY")

llm_name = "gpt-3.5-turbo"

# client = OpenAI(api_key=openai_key)
model = ChatOpenAI(api_key=openai_key, model=llm_name)

# MODEL_NAME = 'llama-3.1-8b-instant'
# model = ChatGroq(model_name=MODEL_NAME, api_key=os.getenv('GROQ_API_KEY'))


class State(TypedDict):
    messages: Annotated[list, add_messages]


# Create tools
search_tool = TavilySearchResults(max_results=2)
tools = [search_tool]
# res = tool.invoke("What is the capital of the UK?")
# print(res)

# Add tools to the model
model_with_tools = model.bind_tools(tools)

# res = model_with_tools.invoke("What is a node in LangGraph?")
# print(res)

import json
from langchain_core.messages import ToolMessage


class BasicToolNode:
    """A node that runs the tools request in the last AI Message"""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No messages found in input.")


def bot(state: State):
    print(state['messages'])
    return {"messages": [model_with_tools.invoke(state['messages'])]}


graph_builder = StateGraph(State)

# Add node
graph_builder.add_node('bot', bot)

# Set entry point
graph_builder.set_entry_point('bot')

# Set finish point
graph_builder.set_finish_point('bot')

# Compile the graph
graph = graph_builder.compile()


# Use the created graph
# res = graph.invoke({"messages": ["Hello, how are you?"]})
# print(res["messages"])

# while True:
#     user_input = input("User: ")
#     if user_input.lower() in ["quit", "exit", "q"]:
#         print("Goodbye!")
#         break
#     for event in graph.stream({"messages": ("user", user_input)}):
#         for value in event.values():
#             print(f"Assistant: ${value["messages"][-1].content}")

