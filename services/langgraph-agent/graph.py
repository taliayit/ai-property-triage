"""planner -> tool_executor (conditional) -> synthesiser. See README."""

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph

from config import GEMINI_API_KEY, GEMINI_MODEL_NAME, GEMINI_TEMPERATURE
from tools import TOOLS

TOOLS_BY_NAME = {t.name: t for t in TOOLS}


class AgentState(TypedDict):
    query: str
    messages: Annotated[list[AnyMessage], operator.add]
    tools_used: Annotated[list[str], operator.add]
    reasoning_steps: Annotated[list[str], operator.add]
    answer: str


def initial_state(query: str) -> AgentState:
    return {"query": query, "messages": [], "tools_used": [], "reasoning_steps": [], "answer": ""}


def get_llm(bind_tools: bool = False):
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL_NAME,
        google_api_key=GEMINI_API_KEY,
        temperature=GEMINI_TEMPERATURE,
    )
    return llm.bind_tools(TOOLS) if bind_tools else llm


PLANNER_SYSTEM_PROMPT = (
    "You are a real estate assistant that answers questions about property "
    "listings and room photos. You have tools available - call whichever "
    "ones (zero, one, or more) are actually needed to answer the user's "
    "question; read each tool's description carefully, they cover "
    "different things. Don't call a tool the question doesn't need - "
    "general advice questions need no tools at all. If the question refers "
    "to multiple photos, call the photo tool once per photo URL."
)

SYNTHESISER_SYSTEM_PROMPT = (
    "You are a real estate assistant. Using the conversation so far "
    "(including any tool results), write a clear, direct answer to the "
    "user's original question. Ground every specific claim in the tool "
    "results provided - don't invent details the tools didn't return. If "
    "no tools were called, answer from general knowledge."
)


def planner_node(state: AgentState) -> dict:
    human_message = HumanMessage(content=state["query"])
    response = get_llm(bind_tools=True).invoke(
        [SystemMessage(content=PLANNER_SYSTEM_PROMPT), human_message]
    )

    tool_calls = response.tool_calls or []
    step = (
        f"Planner decided to call: {', '.join(c['name'] for c in tool_calls)}"
        if tool_calls
        else "Planner decided no tools are needed for this query."
    )
    return {"messages": [human_message, response], "reasoning_steps": [step]}


def tool_executor_node(state: AgentState) -> dict:
    last_message = state["messages"][-1]
    tool_messages = []
    tools_used = []
    steps = []

    for call in last_message.tool_calls:
        tool_fn = TOOLS_BY_NAME[call["name"]]
        result = tool_fn.invoke(call["args"])
        tool_messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
        tools_used.append(call["name"])
        steps.append(f"Called {call['name']}({call['args']}) -> {str(result)[:200]}")

    return {"messages": tool_messages, "tools_used": tools_used, "reasoning_steps": steps}


def extract_text(content) -> str:
    """Normalizes response.content to plain text (this Gemini version
    returns a list of content blocks instead of a string)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content)


def synthesiser_node(state: AgentState) -> dict:
    if state["tools_used"]:
        messages = [SystemMessage(content=SYNTHESISER_SYSTEM_PROMPT)] + state["messages"]
    else:
        # Gemini rejects a request ending on a bare assistant turn, so
        # synthesise from the query directly rather than the planner's message.
        messages = [
            SystemMessage(content=SYNTHESISER_SYSTEM_PROMPT),
            HumanMessage(content=state["query"]),
        ]
    response = get_llm().invoke(messages)
    return {
        "answer": extract_text(response.content),
        "reasoning_steps": ["Synthesiser combined results into a final answer."],
    }


def route_after_planner(state: AgentState) -> str:
    last_message = state["messages"][-1]
    return "tool_executor" if getattr(last_message, "tool_calls", None) else "synthesiser"


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("planner", planner_node)
    graph.add_node("tool_executor", tool_executor_node)
    graph.add_node("synthesiser", synthesiser_node)

    graph.set_entry_point("planner")
    graph.add_conditional_edges(
        "planner", route_after_planner, {"tool_executor": "tool_executor", "synthesiser": "synthesiser"}
    )
    graph.add_edge("tool_executor", "synthesiser")
    graph.add_edge("synthesiser", END)

    return graph.compile()
