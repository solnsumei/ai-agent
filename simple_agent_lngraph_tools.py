import os
from typing import TypedDict, Annotated

from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

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
tools = [search_tool]
# res = tool.invoke("What is the capital of the UK?")
# print(res)

# Add tools to the model
model_with_tools = model.bind_tools(tools)

# res = model_with_tools.invoke("What is a node in LangGraph?")
# print(res)

from langchain_core.messages import BaseMessage
from langgraph.prebuilt import ToolNode, tools_condition


def bot(state: State):
    print(state['messages'])
    return {"messages": [model_with_tools.invoke(state['messages'])]}


tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "bot",
    tools_condition,
)

# Add node
graph_builder.add_node('bot', bot)

# Set entry point
graph_builder.set_entry_point('bot')

# from langgraph.checkpoint.sqlite import SqliteSaver
#
# with SqliteSaver.from_conn_string(":memory:") as memory:
#     graph = graph_builder.compile(checkpointer=memory, interrupt_before=["tools"])
#
#     # graph_builder.set_finish_point('bot')
#     config = {
#         "configurable": {"thread_id": 1}
#     }
#
#     user_input = "I'm learning AI. could you do some research on it for me?"
#
#     events = graph.stream(
#         {"messages": [("user", user_input)]}, config, stream_mode="values"
#     )
#
#     for event in events:
#         event["messages"][-1].pretty_print()
#
#     snapshot = graph.get_state(config)
#     next_step = snapshot.next
#
#     print("====>>>", next_step)
#
#     existing_message = snapshot.values["messages"][-1]
#     all_tools = existing_message.tool_calls
#
#     print("tools to be called::", all_tools)
#
#     events = graph.stream(None, config, stream_mode="values")
#
#     for event in events:
#         if "messages" in event:
#             event["messages"][-1].pretty_print()


from langgraph.checkpoint.memory import MemorySaver
memory = MemorySaver()

graph = graph_builder.compile(checkpointer=memory)
# graph = graph_builder.compile(checkpointer=memory, interrupt_before=["tools"])

# graph_builder.set_finish_point('bot')
config = {
    "configurable": {"thread_id": 1}
}

user_input = "Hello, my name is Solomon, I'm a happily married man with a godly wife and 3 godly children."

events = graph.stream(
    {"messages": [("user", user_input)]}, config, stream_mode="values"
)

for event in events:
    event["messages"][-1].pretty_print()

# snapshot = graph.get_state(config)
# next_step = snapshot.next
#
# print("====>>>", next_step)
#
# existing_message = snapshot.values["messages"][-1]
# all_tools = existing_message.tool_calls
#
# print("tools to be called::", all_tools)
#
# events = graph.stream(None, config, stream_mode="values")
#
# for event in events:
#     if "messages" in event:
#         event["messages"][-1].pretty_print()


# Use the created graph
# res = graph.invoke({"messages": ["Hello, how are you?"]})
# print(res["messages"])
#
# while True:
#     user_input = input("User: ")
#     if user_input.lower() in ["quit", "exit", "q"]:
#         print("Goodbye!")
#         break
#     for event in graph.stream({"messages": ("user", user_input)}):
#         for value in event.values():
#             if isinstance(value["messages"][-1], BaseMessage):
#                 print(f"Assistant: {value["messages"][-1].content}")

while True:
    user_input = input("User: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break
    events = graph.stream({"messages": [("user", user_input)]}, config, stream_mode="values")
    for event in events:
        if "messages" in event:
            print(f"Assistant: {event["messages"][-1].content}")

