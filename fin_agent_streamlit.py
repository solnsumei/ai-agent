import enum
import os
import json
import operator
from io import StringIO
import pandas as pd
from typing import TypedDict, Annotated, List

from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.pydantic_v1 import BaseModel

from langgraph.graph import add_messages, StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage
# from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")
tavily_key = os.getenv("TAVILY_API_KEY")

llm_name = "gpt-3.5-turbo"

# client = OpenAI(api_key=openai_key)
# model = ChatOpenAI(api_key=openai_key, model=llm_name)

MODEL_NAME = 'llama-3.1-8b-instant'
model = ChatGroq(model_name=MODEL_NAME, api_key=os.getenv('GROQ_API_KEY'))

# Initialize memory
memory = MemorySaver()
from tavily import TavilyClient

tavily_client = TavilyClient(api_key=tavily_key)


# tavily = TavilySearchResults(max_results=2)


class AgentState(TypedDict):
    task: str
    competitors: List[str]
    csv_file: str
    financial_data: str
    analysis: str
    competitor_data: str
    comparison: str
    feedback: str
    report: str
    content: List[str]
    revision_number: int
    max_revisions: int


class Queries(BaseModel):
    queries: List[str]


# Define the prompts for each node - IMPROVE AS NEEDED
GATHER_FINANCIALS_PROMPT = """You are an expert financial analyst. Gather the financial data for the given company.
 Provide detailed financial data."""
ANALYZE_DATA_PROMPT = """You are an expert financial analyst. Analyze the provided financial data and provide
 detailed insights and analysis."""
RESEARCH_COMPETITORS_PROMPT = """You are a researcher tasked with providing information about similar companies for 
performance comparison. Generate a list of search queries to gather relevant information.
 Only generate 3 queries max."""
COMPETE_PERFORMANCE_PROMPT = """You are an expert financial analyst. Compare the financial performance of the given
 company with its competitors based on the provided data.
**MAKE SURE TO INCLUDE THE NAMES OF THE COMPETITORS IN THE COMPARISON.**"""
FEEDBACK_PROMPT = """You are a reviewer. Provide detailed feedback and critique for the provided financial comparison
 report. Include any additional information or revisions needed."""
WRITE_REPORT_PROMPT = """You are a financial report writer. Write a comprehensive financial report based on the
 analysis, competitor research, comparison, and feedback provided."""
RESEARCH_CRITIQUE_PROMPT = """You are a researcher tasked with providing information to address the provided critique.
 Generate a list of search queries to gather relevant information. Only generate 3 queries max."""


def gather_financials_node(state: AgentState):
    # Read the csv file into a pandas Dataframe
    csv_file = state["csv_file"]

    df = pd.read_csv(StringIO(csv_file))

    financial_data_str = df.to_string(index=False)

    # Combine the financial data string with the task
    combined_content = (
        f"{state['task']}\n\nHere is the financial data:\n\n{financial_data_str}"
    )

    messages = [
        SystemMessage(content=GATHER_FINANCIALS_PROMPT),
        HumanMessage(content=combined_content)
    ]

    response = model.invoke(messages)
    return {"financial_data": response.content}


def analyze_data_node(state: AgentState):
    messages = [
        SystemMessage(content=ANALYZE_DATA_PROMPT),
        HumanMessage(content=state["financial_data"])
    ]

    response = model.invoke(messages)
    return {"analysis": response.content}


def research_competitors_node(state: AgentState):
    content = state.get("content", [])
    for competitor in state["competitors"]:
        queries = model.with_structured_output(Queries).invoke(
            [
                SystemMessage(content=RESEARCH_COMPETITORS_PROMPT),
                HumanMessage(content=competitor)
            ]
        )
        for q in queries.queries:
            response = tavily_client.search(query=q, max_results=2)
            for r in response["results"]:
                content.append(r["content"])
    return {"content": content}


def compare_performance_node(state: AgentState):
    content = "\n\n".join(state["content"] or [])
    user_message = HumanMessage(
        content=f"{state['task']}\n\nHere is the financial analysis:\n\n{state['analysis']}"
    )
    messages = [
        SystemMessage(content=COMPETE_PERFORMANCE_PROMPT.format(content=content)),
        user_message
    ]

    response = model.invoke(messages)
    return {
        "comparison": response.content,
        "revision_number": state.get("revision_number", 1) + 1,
    }


def research_critique_node(state: AgentState):
    queries = model.with_structured_output(Queries).invoke(
        [
            SystemMessage(content=RESEARCH_CRITIQUE_PROMPT),
            HumanMessage(content=state["feedback"]),
        ]
    )
    content = state["content"] or []
    for q in queries.queries:
        response = tavily_client.search(query=q, max_results=2)
        for r in response["results"]:
            content.append(r["content"])
    return {"content": content}


def collect_feedback_node(state: AgentState):
    messages = [
        SystemMessage(content=FEEDBACK_PROMPT),
        HumanMessage(content=state["comparison"]),
    ]
    response = model.invoke(messages)
    return {"feedback": response.content}


def write_report_node(state: AgentState):
    messages = [
        SystemMessage(content=WRITE_REPORT_PROMPT),
        HumanMessage(content=state["comparison"]),
    ]
    response = model.invoke(messages)
    return {"report": response.content}


def should_continue(state: AgentState):
    if state["revision_number"] > state["max_revisions"]:
        return END
    return "collect_feedback"


class AgentNode(enum.StrEnum):
    GATHER_FINANCIALS = "gather_financials"
    ANALYZE_DATA = "analyze_data"
    RESEARCH_COMPETITORS = "research_competitors"
    COMPARE_PERFORMANCE = "compare_performance"
    COLLECT_FEEDBACK = "collect_feedback"
    RESEARCH_CRITIQUE = "research_critique"
    WRITE_REPORT = "write_report"


builder = StateGraph(AgentState)
builder.add_node(AgentNode.GATHER_FINANCIALS, gather_financials_node)
builder.add_node(AgentNode.ANALYZE_DATA, analyze_data_node)
builder.add_node(AgentNode.RESEARCH_COMPETITORS, research_competitors_node)
builder.add_node(AgentNode.COMPARE_PERFORMANCE, compare_performance_node)
builder.add_node(AgentNode.COLLECT_FEEDBACK, collect_feedback_node)
builder.add_node(AgentNode.RESEARCH_CRITIQUE, research_critique_node)

builder.add_node(AgentNode.WRITE_REPORT, write_report_node)

builder.set_entry_point(AgentNode.GATHER_FINANCIALS)

builder.add_conditional_edges(
    AgentNode.COMPARE_PERFORMANCE,
    should_continue,
    {END: END, "collect_feedback": AgentNode.COLLECT_FEEDBACK}
)

builder.add_edge(AgentNode.GATHER_FINANCIALS, AgentNode.ANALYZE_DATA)
builder.add_edge(AgentNode.ANALYZE_DATA, AgentNode.RESEARCH_COMPETITORS)
builder.add_edge(AgentNode.RESEARCH_COMPETITORS, AgentNode.COMPARE_PERFORMANCE)
builder.add_edge(AgentNode.COLLECT_FEEDBACK, AgentNode.RESEARCH_CRITIQUE)
builder.add_edge(AgentNode.RESEARCH_CRITIQUE, AgentNode.COMPARE_PERFORMANCE)
builder.add_edge(AgentNode.COMPARE_PERFORMANCE, AgentNode.WRITE_REPORT)

graph = builder.compile(checkpointer=memory)


# =============== Streamlit UI =============
import streamlit as st


def main():
    st.title("Financial Performance Reporting")

    task = st.text_input(
        "Enter the task:",
        "Analyze the financial performance of our company compared to our competitors"
    )

    competitors = st.text_area("Enter competitor names (one per line)").split("\n")
    max_revisions = st.number_input("Max Revisions", min_value=1, value=2, max_value=3)
    uploaded_file = st.file_uploader(
        "Upload a CSV file with the company's financial data", type=["csv"]
    )

    if st.button("Start Analysis") and uploaded_file is not None:
        csv_data = uploaded_file.getvalue().decode("utf-8")

        initial_state = {
            "task": task,
            "competitors": [comp.strip() for comp in competitors if comp.strip()],
            "csv_file": csv_data,
            "max_revisions": max_revisions,
            "revision_number": 1,
        }

        thread = {"configurable": {"thread_id": "1"}}

        final_state = None
        for s in graph.stream(initial_state, thread):
            st.write(s)
            final_state = s

        if final_state and "report" in final_state:
            st.subheader("Final Report")
            st.write(final_state["report"])


if __name__ == "__main__":
    main()

