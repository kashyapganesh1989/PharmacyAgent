# =============================================================================
# Pharmacy IT Operations Agent — Agent Builder
# Author : Abhinav Rawat · AI Architect · AI Playground
#
# KEY GUARDRAIL PATTERN DEMONSTRATED HERE:
#   High-risk tools (reset_user_password, restart_store_device) are NEVER
#   bound to the agent directly. Instead the agent only ever sees a "guarded"
#   version that QUEUES a human-approval request and tells the agent (truth-
#   fully) that the action has not happened yet. The real action function
#   (agent.tools.execute_reset_user_password / execute_restart_store_device)
#   is only ever called by the Streamlit UI, after a human clicks Approve.
#
#   This is the ❌ DON'T / ✅ DO contrast to call out live in training:
#     ❌ Don't bind a destructive tool straight to the LLM and trust the
#        system prompt to stop it from being misused.
#     ✅ Do make the tool itself incapable of the destructive action —
#        the approval gate is structural, not a suggestion to the model.
# =============================================================================

import uuid

from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from agent.tools import (
    ALL_TOOLS,
    ROLE_TOOL_ACCESS,
    HIGH_RISK_DESCRIPTIONS,
)

SYSTEM_PROMPT = """You are the Pharmacy IT Operations Agent, a helpful internal \
assistant for store associates and IT admins at a retail pharmacy chain.

You help with: searching IT knowledge base articles, checking store system \
status, checking inventory sync status, filing ITSM tickets, escalating to \
on-call engineers, and (IT Admins only) requesting password resets or device \
restarts.

Rules you MUST follow:
- For reset_user_password and restart_store_device: these tools only QUEUE a \
request for human approval. They do NOT execute immediately. Always tell the \
user clearly that the action is pending approval, not completed.
- Never claim an action succeeded unless the tool result says it did.
- Keep answers concise and specific. Cite ticket/system IDs when relevant.
- If you are unsure, search the knowledge base before escalating.
"""


def _queue_approval(tool_name: str, arg_name: str, value: str, pending_approvals: dict) -> str:
    request_id = f"req_{uuid.uuid4().hex[:8]}"
    pending_approvals[request_id] = {
        "tool_name": tool_name,
        "arg_name": arg_name,
        "arg_value": value,
        "status": "pending",
    }
    return (
        f"⏸️ Approval required (request ID: {request_id}) before "
        f"'{tool_name}' can run on '{value}'. A human admin must approve this "
        f"in the UI before it takes effect."
    )


def _make_guarded_reset_password_tool(pending_approvals: dict):
    description = (
        f"{HIGH_RISK_DESCRIPTIONS['reset_user_password']} This action requires "
        f"human approval and will NOT execute immediately — it only queues a request."
    )

    def _run(username: str) -> str:
        """Request a password reset for a store employee. Only queues the
        request for human approval — does not execute immediately."""
        return _queue_approval("reset_user_password", "username", username, pending_approvals)

    return StructuredTool.from_function(func=_run, name="reset_user_password", description=description)


def _make_guarded_restart_device_tool(pending_approvals: dict):
    description = (
        f"{HIGH_RISK_DESCRIPTIONS['restart_store_device']} This action requires "
        f"human approval and will NOT execute immediately — it only queues a request."
    )

    def _run(device_id: str) -> str:
        """Request a remote restart of a store device. Only queues the
        request for human approval — does not execute immediately."""
        return _queue_approval("restart_store_device", "device_id", device_id, pending_approvals)

    return StructuredTool.from_function(func=_run, name="restart_store_device", description=description)


def get_tools_for_role(role: str, pending_approvals: dict):
    """Returns the list of LangChain tools bound for this role — low/medium
    risk tools as-is, high-risk tools wrapped in the approval gate, and only
    the ones the role is allowed to see at all."""
    allowed_names = set(ROLE_TOOL_ACCESS.get(role, []))

    tools = [t for t in ALL_TOOLS if t.name in allowed_names]

    if "reset_user_password" in allowed_names:
        tools.append(_make_guarded_reset_password_tool(pending_approvals))
    if "restart_store_device" in allowed_names:
        tools.append(_make_guarded_restart_device_tool(pending_approvals))

    return tools


def build_agent(api_key: str, role: str, pending_approvals: dict, model: str = "gpt-4o-mini"):
    """Builds an AgentExecutor scoped to the given role. Callbacks (audit
    logging) are passed in per-invocation from the Streamlit app, not baked
    in here, so each turn's trace stays attributable to that turn."""
    tools = get_tools_for_role(role, pending_approvals)

    llm = ChatOpenAI(model=model, temperature=0, api_key=api_key)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        max_iterations=6,
        verbose=False,
        handle_parsing_errors=True,
    )
    return executor, tools
