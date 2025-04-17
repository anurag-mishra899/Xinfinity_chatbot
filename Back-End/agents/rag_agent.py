from langchain.agents import tool, initialize_agent
from langchain import hub
import pandas as pd
from uuid import uuid4
from langgraph.prebuilt import create_react_agent
import faiss
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_core.runnables import Runnable
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import InMemoryDocstore
from langchain.tools.retriever import create_retriever_tool
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain.agents import create_openai_functions_agent, AgentExecutor



### CONFIGS ###
PDF_FILES = ['HOW5220_DG558_XB6_getting_started.pdf', 'RSA_01012025.pdf']
CHUNK_SIZE = 1500
VECTOR_STORE_PATH = "<PATH TO >NBs/vector_store_index"
# llm = ChatOpenAI(model="gpt-4")

### STEP 1: Load and chunk documents ###

# def load_and_chunk_documents(pdf_files, chunk_size):
#     parser = DocumentParser()
#     chunker = DocumentChunk()
#     all_chunks = pd.DataFrame()

#     for pdf in pdf_files:
#         result = parser.extract_text(pdf, type='local')
#         chunk_df = chunker.process_chunk(pd.DataFrame(result), chunk_size)
#         chunk_df['source'] = pdf
#         all_chunks = pd.concat([chunk_df, all_chunks])

#     return all_chunks.reset_index(drop=True)


### STEP 2: Convert DataFrame to LangChain Documents ###
# def dataframe_to_documents(df: pd.DataFrame):
#     documents = []
#     for _, row in df.iterrows():
#         documents.append(
#             Document(
#                 page_content=row["merged_content"],
#                 metadata={
#                     "source": row["source"],
#                     "page_range": row["page_range"],
#                     "chunk_coordinates": row["chunk_coordinates"],
#                 },
#             )
#         )
#     return documents


### STEP 3: Build or Load FAISS Vector Store ###
def build_vector_store(documents, embeddings, save_path):
    dimension = len(embeddings.embed_query("hello world"))
    index = faiss.IndexFlatL2(dimension)
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    ids = [str(uuid4()) for _ in documents]
    vector_store.add_documents(documents, ids=ids)
    vector_store.save_local(save_path)
    return vector_store


def load_vector_store(path, embeddings):
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)


### STEP 4: Create Retrieval Tool ###
def get_retriever_tool(vector_store):
    retriever = vector_store.as_retriever()
    return create_retriever_tool(
        retriever=retriever,
        name="retriever_tool",
        description="For fetching company-related policies or guidelines information.",
    )


### STEP 6: Create RAG Agent ###
def create_rag_agent(llm):
    embeddings = OpenAIEmbeddings()
    vector_store=None
    try:
        vector_store = load_vector_store(VECTOR_STORE_PATH, embeddings)
    except Exception:
        # todo : 
        # If not available, build from scratch
        # chunked_df = load_and_chunk_documents(PDF_FILES, CHUNK_SIZE)
        # documents = dataframe_to_documents(chunked_df)
        # vector_store = build_vector_store(documents, embeddings, VECTOR_STORE_PATH)
        print("did not find vector store at this path")
        vector_store=None
    retriever_tool = get_retriever_tool(vector_store)
    
    rag_react_agent = create_react_agent(model= llm, 
                                        tools= [retriever_tool], 
                                        name= 'rag_agent_executor',
                                        prompt= "You are a RAG agent that answers user queries, you have access to the retriever tool for context")
    return rag_react_agent
    



# --- Step 4: Run the agent (example query)
# rag_agent_executer = create_rag_agent(llm=llm)

# query = "pls share contact info , im a massachusets resident"

# query="what is liability for the damage?"
# response = rag_agent_executer.invoke({"messages": [query]})
# print(response)