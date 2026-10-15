# =============================================================================
# Pharmacy IT Operations Agent — LangGraph orchestration
#
#   START → retrieve → agent ─┬─→ tools ────────┐
#                             ├─→ admin_tools ──┤   (graph PAUSES before
#                             ├─→ not_allowed ──┤    admin_tools until a
#                             └─→ END           │    human approves)
#                  agent ←──────────────────────┘
#
# KEY PATTERNS DEMONSTRATED HERE:
#   - Retrieval is a deterministic node, not a model decision: every turn
#     starts by pulling top-k passages over the relevance threshold.
#   - High-risk tools live in their own node, and the graph is compiled with
#     interrupt_before=["admin_tools"]. The approval gate is structural: the
#     model cannot talk its way past it.
#   - Role allowlist enforced twice: the model is only shown its role's tools,
#     and any call to a tool outside the allowlist is refused in code.
#   - Hop limit: at most MAX_HOPS agent steps per turn, so no endless loops.
#   - State (messages, retrieved passages) is checkpointed to SQLite by
#     thread_id, so a conversation survives an app restart.
# =============================================================================

import sqlite3
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from agent import db, vectorstore
from agent.callbacks import tool_emit
from agent.tools import ALL_TOOLS, HIGH_RISK_TOOLS, ROLE_TOOL_ACCESS, tools_for_role

MAX_HOPS = 6
MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """You are the Pharmacy IT Operations Agent, an internal assistant \
for store associates and IT admins at a retail pharmacy chain.

Grounding rules:
- For questions about policy or procedures, answer ONLY from the CONTEXT passages \
below. Cite the chunk ID of every passage you use in square brackets, exactly as \
written, e.g. [pharmacy_policy.md#2].
- If the CONTEXT does not contain the answer, say that the uploaded documents do \
not cover it. Do not answer from general knowledge.

Tool rules:
- For live stock questions, call search_inventory. If it fails, say stock is \
unknown; never guess a number.
- To reset data: call preview_sql_reset first, then call reset_sql_state in the \
same turn. Do not ask the user to confirm in chat: the app pauses for human \
approval before reset_sql_state runs.
- reset_sql_state, update_user_role and send_password_reset all pause for human \
approval. Only say an action is done if its tool result says ok.
- Never ask for, accept, set or reveal a password.
Keep answers short and specific.

CONTEXT:
{context}"""


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    retrieved: list[dict]
    hops: int


def _conf(config: RunnableConfig) -> dict:
    return (config or {}).get("configurable", {})


def retrieve_node(state: AgentState, config: RunnableConfig) -> dict:
    conf = _conf(config)
    question = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
    result = vectorstore.retrieve(question, conf["api_key"],
                                  min_score=conf.get("min_score", vectorstore.DEFAULT_MIN_SCORE))
    passed = result["passed"]
    top = max([r["score"] for r in passed + result["rejected"]], default=None)
    tool_emit(
        step_type="retrieve", label="Retrieve — vector DB",
        status="ok",
        detail=(f"{len(passed)} passage(s) ≥ {result['min_score']}: {', '.join(r['chunk_id'] for r in passed)}"
                if passed else
                f"no passage ≥ {result['min_score']} (best {top if top is not None else 'n/a'}) → "
                f"answer must say not found"),
    )
    return {"retrieved": passed, "hops": 0}


def agent_node(state: AgentState, config: RunnableConfig) -> dict:
    conf = _conf(config)
    context = "\n\n".join(f"[{r['chunk_id']}]\n{r['text']}" for r in state["retrieved"]) or "NO MATCHING DOCUMENTS"
    llm = ChatOpenAI(model=MODEL, temperature=0, api_key=conf["api_key"], timeout=30, max_retries=2)
    llm = llm.bind_tools(tools_for_role(conf.get("caller_role", "Store Associate")))
    reply = llm.invoke([SystemMessage(SYSTEM_PROMPT.format(context=context)), *state["messages"]], config)
    return {"messages": [reply], "hops": state.get("hops", 0) + 1}


def not_allowed_node(state: AgentState, config: RunnableConfig) -> dict:
    """Refuses tool calls outside the caller's role allowlist, in code."""
    role = _conf(config).get("caller_role", "Store Associate")
    allowed = set(ROLE_TOOL_ACCESS.get(role, []))
    out = []
    for call in state["messages"][-1].tool_calls:
        if call["name"] in allowed:
            content = '{"ok": false, "error": "skipped", "hint": "Another tool in this step was not allowed; retry it alone."}'
        else:
            content = f'{{"ok": false, "error": "not_allowed", "hint": "Role {role} cannot use {call["name"]}."}}'
            tool_emit(step_type="allowlist", label=f"Role allowlist — blocked {call['name']}", status="blocked",
                      detail=f"'{role}' is not allowed to call {call['name']}", risk_tier="high")
        out.append(ToolMessage(content=content, tool_call_id=call["id"], name=call["name"]))
    return {"messages": out}


def route_after_agent(state: AgentState, config: RunnableConfig) -> str:
    last = state["messages"][-1]
    calls = getattr(last, "tool_calls", None) or []
    if not calls:
        return END
    if state.get("hops", 0) >= MAX_HOPS:
        tool_emit(step_type="hop_limit", label="Hop limit reached", status="blocked",
                  detail=f"stopped after {MAX_HOPS} agent steps")
        return END
    allowed = set(ROLE_TOOL_ACCESS.get(_conf(config).get("caller_role", "Store Associate"), []))
    if any(c["name"] not in allowed for c in calls):
        return "not_allowed"
    if any(c["name"] in HIGH_RISK_TOOLS for c in calls):
        return "admin_tools"
    return "tools"


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode([t for t in ALL_TOOLS if t.name not in HIGH_RISK_TOOLS]))
    # High-risk calls arrive here only as a whole step; low-risk calls made in
    # the same step ride along and run after approval too.
    graph.add_node("admin_tools", ToolNode(ALL_TOOLS))
    graph.add_node("not_allowed", not_allowed_node)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "agent")
    graph.add_conditional_edges("agent", route_after_agent, ["tools", "admin_tools", "not_allowed", END])
    graph.add_edge("tools", "agent")
    graph.add_edge("admin_tools", "agent")
    graph.add_edge("not_allowed", "agent")

    db.init_db()
    conn = sqlite3.connect(db.CHECKPOINT_DB_PATH, check_same_thread=False)
    return graph.compile(checkpointer=SqliteSaver(conn), interrupt_before=["admin_tools"])


def deny_pending(graph, config: RunnableConfig, decided_by: str):
    """Answers every pending high-risk call with a 'denied' tool result, as if
    admin_tools had run, so the agent can continue and explain."""
    state = graph.get_state(config)
    calls = state.values["messages"][-1].tool_calls
    msgs = [
        ToolMessage(content=f'{{"ok": false, "error": "denied", "hint": "Denied by {decided_by}. Nothing was changed."}}',
                    tool_call_id=c["id"], name=c["name"])
        for c in calls
    ]
    graph.update_state(config, {"messages": msgs}, as_node="admin_tools")


def pending_calls(graph, config: RunnableConfig) -> list[dict]:
    state = graph.get_state(config)
    if "admin_tools" not in (state.next or ()):
        return []
    return state.values["messages"][-1].tool_calls


def thread_message_count(graph, config: RunnableConfig) -> int:
    state = graph.get_state(config)
    return len((state.values or {}).get("messages", []))
