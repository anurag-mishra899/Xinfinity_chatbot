from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, PromptTemplate

from langgraph.prebuilt import create_react_agent


# --- Step 1: Set up database + LLM


# print(db.dialect)
# print(db.get_usable_table_names())

def get_sql_tools(llm,DB):
    """
    Returns SQL tools using LangChain's SQLDatabaseToolkit.
    """
    toolkit = SQLDatabaseToolkit(db=DB, llm=llm)
    return toolkit.get_tools()


def get_custom_sql_prompt(dialect="SQLite", top_k=10):
    """
    Loads and customizes the SQL agent system prompt.
    """
    # prompt_template = hub.pull("langchain-ai/sql-agent-system-prompt")
    

    # Set your custom system prompt
    system_prompt = """
        You are an intelligent assistant helping users explore a customer billing system.
        Given an input question, create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
        Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results.
        You can order the results by a relevant column to return the most interesting examples in the database.
        Never query for all the columns from a specific table, only ask for the relevant columns given the question.
        You have access to tools for interacting with the database.
        Only use the below tools. Only use the information returned by the below tools to construct your final answer.
        You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

        DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

        To start you should ALWAYS look at the tables in the database to see what you can query.
        Do NOT skip this step.
        Then you should query the schema of the most relevant tables. 
        Before executing a query, make sure you have:
        1. The correct **table** to query.
        2. All **mandatory fields** for that table.
        3. Clear disambiguation if needed (e.g., same name, multiple services).

        It is important that You ask follow-up questions to gather missing fields based on this schema:
        - customer_table needs customer_id.
        - customer_service_table needs customer_id + service_id.
        - billing_table needs customer_id + service_id (+ optional billing_date, status, plan_name).

        When ready, construct a SQL-like query or structured representation to fetch the data."""
    prompt_template= ChatPromptTemplate(messages=[SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=['dialect', 'top_k'], input_types={}, partial_variables={},template=system_prompt), additional_kwargs={})])

    return prompt_template.format(dialect=dialect, top_k=top_k)

def create_sql_agent(llm, tools, system_message):
    """
    Creates and returns a LangGraph SQL agent.
    """
    return create_react_agent(
        model=llm,
        tools=tools,
        name="sql_agent",
        prompt=system_message
    )


# --- Step 2: Load tools and system prompt
# tools = get_sql_tools(db=db, llm=llm)
# system_prompt = get_custom_sql_prompt(dialect="SQLite", top_k=10)

# # --- Step 3: Create SQL agent
# sql_agent_executor = create_sql_agent(llm=llm, tools=tools, system_message=system_prompt)

# --- Step 4: Run the agent (example query)
# query = "How much did I spend in the last 6 months? my id is 13"
# response = sql_agent_executor.invoke({"messages": [query]})
# print(response)