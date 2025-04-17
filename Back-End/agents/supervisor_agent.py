from langgraph_supervisor import create_supervisor
from .sql_agent import get_sql_tools, get_custom_sql_prompt,create_sql_agent
from .rag_agent import create_rag_agent
from langchain_community.utilities import SQLDatabase

from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4.1-nano-2025-04-14")

DB = SQLDatabase.from_uri("sqlite:///<PATH_TO_XFINITY_AGENT.DB>")


def set_up_sql_agent(llm=llm):
    tools = get_sql_tools( llm=llm,DB=DB)
    system_prompt = get_custom_sql_prompt(dialect="SQLite", top_k=10)
    sql_agent = create_sql_agent(llm=llm, tools=tools, system_message=system_prompt)
    return sql_agent


def set_up_rag_agent(llm=llm):
    rag_agent = create_rag_agent(llm=llm)
    return rag_agent


def create_workflow():
    sql_agent=set_up_sql_agent(llm=llm)
    rag_agent=set_up_rag_agent(llm=llm)

    workflow = create_supervisor(
        agents=[sql_agent, rag_agent],
        model=llm,
        prompt=(
    
            "Role: You are an Xfinity customer support team supervisor, managing 2 kinds of tasks : 1. a customer billing sql analysis as well as a math expert. "
            "2. Answering questions related to Xfinitie's policy and guidelines. "
            "For customer billing realted questions, leverage the sql_agent_executor. "
            "For answering queries related to company's policies and guidelines, use rag_agent_executor to cascade rag_agent_executor's output to user without rephrasing it."
            "Avoid mentioning any agent name in your response."
            "Some questions may need you to interact with both tasks, do it accordingly."
        ),
        supervisor_name="Xfinity",
    )
    app = workflow.compile(name="XfinityTest")
    return app
