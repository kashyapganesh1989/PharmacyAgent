# =============================================================================
# Pharmacy IT Operations Agent — Tools
#
# Every tool has: one job, a typed input schema, a docstring written for the
# model, and a structured JSON result ({"ok": true/false, ...}) that the model
# can act on — including whether a retry makes sense.
#
# Each tool is tagged with a RISK TIER. The tier drives three things:
#   1. Which role can see/use the tool at all   (ROLE_TOOL_ACCESS)
#   2. Which graph node runs it: high-risk tools run in the "admin_tools"
#      node, and the graph PAUSES before that node until a human approves
#      (see agent/graph.py — the approval gate is structural, not a prompt).
#   3. Defense in depth: high-risk tools re-check the caller's role from the
#      session config, never from anything the model passes in.
#
# External systems are mocks (agent/mock_services.py) or the local SQLite DB.
# =============================================================================

import hashlib
import json
import secrets
from datetime import datetime, timedelta
from typing import Literal

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from tenacity import RetryError, retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from agent import db, mock_services
from agent.callbacks import tool_emit
from agent.mock_services import TransientError

# -----------------------------------------------------------------------------
# Risk tiers — the single source of truth every other module reads from.
# -----------------------------------------------------------------------------
TOOL_RISK = {
    "search_inventory": "low",
    "check_system_status": "low",
    "list_users": "low",
    "preview_sql_reset": "low",
    "create_itsm_ticket": "medium",
    "reset_sql_state": "high",
    "update_user_role": "high",
    "send_password_reset": "high",
}
HIGH_RISK_TOOLS = {name for name, risk in TOOL_RISK.items() if risk == "high"}

ROLE_TOOL_ACCESS = {
    "Store Associate": ["search_inventory", "check_system_status", "create_itsm_ticket"],
    "IT Admin": list(TOOL_RISK.keys()),
}

# What the approval card says about each high-risk tool.
HIGH_RISK_DESCRIPTIONS = {
    "reset_sql_state": "Deletes all rows in the scope's tables and restores the seed data.",
    "update_user_role": "Changes a user's role, and with it what they can access.",
    "send_password_reset": "Issues a one-time password reset link to the user's email.",
}


def _ok(**data) -> str:
    return json.dumps({"ok": True, **data})


def _err(error: str, retryable: bool = False, hint: str = "") -> str:
    out = {"ok": False, "error": error, "retryable": retryable}
    if hint:
        out["hint"] = hint
    return json.dumps(out)


def _caller(config: RunnableConfig) -> tuple[str, str]:
    conf = (config or {}).get("configurable", {})
    return conf.get("caller_role", "Store Associate"), conf.get("caller_user", "unknown")


# -----------------------------------------------------------------------------
# LOW RISK — read-only
# -----------------------------------------------------------------------------
class InventoryQuery(BaseModel):
    sku: str = Field(description="Product SKU, for example 'AMOX-500'")
    store_id: str = Field(description="Store number, for example '1042'")


def _log_retry(retry_state):
    tool_emit(step_type="retry", label=f"Retry — search_inventory attempt {retry_state.attempt_number} failed",
              status="retry", detail=f"{retry_state.outcome.exception()} · backing off before the next attempt",
              risk_tier="low")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, min=0.5, max=2),
       retry=retry_if_exception_type(TransientError), before_sleep=_log_retry)
def _query_inventory(sku: str, store_id: str) -> dict:
    return mock_services.INVENTORY.query(sku, store_id)


@tool(args_schema=InventoryQuery)
def search_inventory(sku: str, store_id: str) -> str:
    """Look up live stock for one SKU at one store via the inventory search
    service. Use only when the user asks about current availability or stock."""
    try:
        result = _query_inventory(sku, store_id)
    except RetryError as e:
        tool_emit(step_type="fallback", label="Fallback — inventory search unavailable", status="blocked",
                  detail=f"3 attempts failed ({e.last_attempt.exception()}); telling the user stock is unknown",
                  risk_tier="low")
        return _err("inventory_search_unavailable", retryable=False,
                    hint="Tell the user current stock is unknown right now. Do not guess a number.")
    if not result["found"]:
        return _err("not_found", hint=f"No inventory record for {result['sku']} at store {result['store_id']}.")
    return _ok(source="inventory-search (mock)", **result)


MOCK_SYSTEMS = {
    "POS-1042": {"name": "POS Terminal (Store #1042)", "status": "degraded", "detail": "Intermittent freeze on checkout screen"},
    "PHARM-PRINT-07": {"name": "Pharmacy Label Printer #07", "status": "offline", "detail": "No response — possible roll misfeed"},
    "WIFI-STORE-1042": {"name": "Store #1042 Wi-Fi", "status": "healthy", "detail": "All access points online"},
    "RX-DISPENSE-SYS": {"name": "Rx Dispensing System", "status": "healthy", "detail": "All automation cells reporting normal"},
}


@tool
def check_system_status(system_id: str) -> str:
    """Check the live status of a store system by its ID, e.g. 'POS-1042',
    'PHARM-PRINT-07', 'WIFI-STORE-1042', 'RX-DISPENSE-SYS'."""
    sys = MOCK_SYSTEMS.get(system_id.upper())
    if not sys:
        return _err("unknown_system", hint=f"Known systems: {', '.join(MOCK_SYSTEMS)}")
    return _ok(system_id=system_id.upper(), **sys)


@tool
def list_users() -> str:
    """List store-system users and their current roles."""
    return _ok(users=db.list_users())


class ResetScope(BaseModel):
    scope: Literal["tickets", "users", "all_demo_data"] = Field(
        description="Which demo data to reset: 'tickets', 'users' (also clears password-reset tokens), "
                    "or 'all_demo_data' (tickets, users, reset tokens and inventory)."
    )


@tool(args_schema=ResetScope)
def preview_sql_reset(scope: str) -> str:
    """Dry run of a database reset: reports how many rows each table in the
    scope holds now, without changing anything. Always call this before
    reset_sql_state, and show the counts to the user."""
    return _ok(dry_run=True, scope=scope, rows_now=db.preview_reset(scope),
               note="Nothing was changed. reset_sql_state would delete these rows and restore seed data.")


# -----------------------------------------------------------------------------
# MEDIUM RISK — writes, but non-destructive
# -----------------------------------------------------------------------------
class TicketArgs(BaseModel):
    summary: str = Field(description="One-line description of the problem")
    priority: Literal["Low", "Medium", "High"] = "Medium"


@tool(args_schema=TicketArgs)
def create_itsm_ticket(summary: str, priority: str = "Medium") -> str:
    """File an IT support ticket for a store problem."""
    ticket_id = db.create_ticket(summary, priority)
    return _ok(ticket_id=f"IT-{ticket_id}", priority=priority, summary=summary)


# -----------------------------------------------------------------------------
# HIGH RISK — destructive / auth-changing. These run only after a human
# approves, in the graph's admin_tools node.
# -----------------------------------------------------------------------------
def _require_admin(config: RunnableConfig, action: str):
    role, user = _caller(config)
    allowed = role == "IT Admin"
    tool_emit(step_type="auth_check", label=f"Authorization — {action}",
              status="ok" if allowed else "blocked",
              detail=f"caller {user} has session role '{role}'" + ("" if allowed else " — admin required"),
              risk_tier="high")
    return None if allowed else _err("forbidden", hint="Only an IT Admin session can do this.")


@tool(args_schema=ResetScope)
def reset_sql_state(scope: str, config: RunnableConfig) -> str:
    """Reset demo data in the SQL database for one scope: deletes the rows and
    restores the seed data. Destructive: requires human approval. Call
    preview_sql_reset first."""
    denied = _require_admin(config, "reset_sql_state")
    if denied:
        return denied
    deleted = db.reset_scope(scope)
    return _ok(scope=scope, deleted_rows=deleted, restored="seed data")


class RoleArgs(BaseModel):
    username: str = Field(description="Username, for example 'pat.lee'")
    new_role: Literal["store_associate", "pharmacist", "it_admin"]


@tool(args_schema=RoleArgs)
def update_user_role(username: str, new_role: str, config: RunnableConfig) -> str:
    """Change a user's role. Requires human approval."""
    denied = _require_admin(config, "update_user_role")
    if denied:
        return denied
    user = db.get_user(username)
    if user is None:
        return _err("unknown_user", hint="Call list_users to see valid usernames.")
    db.set_user_role(username, new_role)
    return _ok(username=username, old_role=user["role"], new_role=new_role)


class UsernameArgs(BaseModel):
    username: str = Field(description="Username, for example 'pat.lee'")


@tool(args_schema=UsernameArgs)
def send_password_reset(username: str, config: RunnableConfig) -> str:
    """Email the user a one-time password reset link that expires in 15
    minutes. Never accept, set or reveal a password yourself. Requires
    human approval."""
    denied = _require_admin(config, "send_password_reset")
    if denied:
        return denied
    user = db.get_user(username)
    if user is None:
        return _err("unknown_user", hint="Call list_users to see valid usernames.")
    token = secrets.token_urlsafe(16)
    expires = (datetime.now() + timedelta(minutes=15)).strftime("%Y-%m-%d %H:%M:%S")
    db.store_reset_token(username, hashlib.sha256(token.encode()).hexdigest(), expires)
    mock_services.MAILBOX.send(
        to=user["email"], subject="Your password reset link",
        body=f"Reset your password: https://intranet.example/reset?token={token} (expires {expires})",
    )
    # The token goes to the mailbox only — never back to the model.
    return _ok(username=username, sent_to="the user's email on file", expires=expires)


ALL_TOOLS = [
    search_inventory, check_system_status, list_users, preview_sql_reset,
    create_itsm_ticket, reset_sql_state, update_user_role, send_password_reset,
]


def tools_for_role(role: str):
    allowed = set(ROLE_TOOL_ACCESS.get(role, []))
    return [t for t in ALL_TOOLS if t.name in allowed]
