import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from config import GEMINI_API_KEY
from graph import build_graph, initial_state

logger = logging.getLogger(__name__)

agent = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent
    if GEMINI_API_KEY:
        agent = build_graph()
    else:
        logger.warning("GEMINI_API_KEY is not set. Set it before calling /agent/run.")
    yield


app = FastAPI(title="LangGraph Agent", lifespan=lifespan)


class AgentRequest(BaseModel):
    query: str


@app.post("/agent/run")
def run_agent(request: AgentRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")

    if agent is None:
        raise HTTPException(status_code=503, detail="GEMINI_API_KEY is not set on this service.")

    result = agent.invoke(initial_state(request.query))

    return {
        "answer": result["answer"],
        "tools_used": result["tools_used"],
        "reasoning_steps": result["reasoning_steps"],
    }
