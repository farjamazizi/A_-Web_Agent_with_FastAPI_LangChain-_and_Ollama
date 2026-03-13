from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_community.chat_models import ChatOllama
from langchain.agents import initialize_agent, Tool, AgentType
from ddgs import DDGS

from langchain.tools import tool

#-------------TOOLS-------------
def search_web(query: str, max_results: int =5) -> str:
    """Return the top max_results web search results for query as a single formatted string. Uses
     DuckDuck's API"""
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            results.append(f"- {r['title']} - {r['href']}")
    return "\n".join(results)

@tool
def web_search(query: str) -> str:
    """Search the web for query and return concise results."""
    return search_web(query)

#-------------AGENT-------------
MODEL = "llama3.2:3b"
llm = ChatOllama(model=MODEL, temperature=0.1)
tools = [web_search]
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

#-------------FASTAPI-------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173"
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask(req: AskRequest):
    """Proxy the question to the LangChain agent and return its answer."""
    result = agent.invoke({"input": req.question})
    return {"answer": result["output"]}
