from fastapi import FastAPI, Header, Request
from pydantic import BaseModel
from typing import Optional
from agents import supervisor_agent 
from dotenv import load_dotenv

app = FastAPI()

load_dotenv()

graph = supervisor_agent.create_workflow()

class QueryInput(BaseModel):
    query: str

@app.post("/api/v1/generate-stream/")
async def generate_stream(query: QueryInput, x_thread_id: Optional[str] = Header(None)):
    qq= query.query
    thread_id = x_thread_id or "no-thread-id"
    
    answer=graph.invoke(input={
    "messages": [
        {
            "role": "user",
            "content": qq
        }
    ]
    },
    config={"configurable": {"thread_id": thread_id}})
    return answer
    