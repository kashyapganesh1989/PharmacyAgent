# =============================================================================
# Pharmacy IT Operations Agent — Tools
# Author : Abhinav Rawat · AI Architect · AI Playground
#
# Every tool below is a MOCK (no real backend). That is intentional: this is
# a TRAINING DEMO about how to wire guardrails, approvals, audit and rate
# limiting around tool-calling agents — not a production ITSM integration.
#
# Each tool is tagged with a RISK TIER. The tier drives two things:
#   1. Which role can see/use the tool at all   (see ROLE_TOOL_ACCESS)
#   2. Whether it needs human approval before it actually executes
#      (see HIGH_RISK_TOOLS + the guarded-tool wrapper in agent.py)
# =============================================================================

import random
import uuid
from datetime import datetime, timedelta

from langchain_core.tools import tool

# -----------------------------------------------------------------------------
# Risk tiers — the single source of truth every other module reads from.
# -----------------------------------------------------------------------------
TOOL_RISK = {
    "search_knowledge_base": "low",
    "check_system_status": "low",
    "lookup_store_inventory_sync_status": "low",
    "create_itsm_ticket": "medium",
    "escalate_to_oncall": "medium",
    "reset_user_password": "high",
    "restart_store_device": "high",
}

HIGH_RISK_TOOLS = {name for name, risk in TOOL_RISK.items() if risk == "high"}

# -----------------------------------------------------------------------------
# Role-based tool access — the DEMO's guardrail #1: not every role can even
# see every tool. A Store Associate never gets reset_user_password bound to
# their agent at all; an IT Admin does (but it still needs approval).
# -----------------------------------------------------------------------------
ROLE_TOOL_ACCESS = {
    "Store Associate": [
        "search_knowledge_base",
        "check_system_status",
        "create_itsm_ticket",
        "lookup_store_inventory_sync_status",
        "escalate_to_oncall",
    ],
    "IT Admin": [
        "search_knowledge_base",
        "check_system_status",
        "create_itsm_ticket",
        "lookup_store_inventory_sync_status",
        "escalate_to_oncall",
        "reset_user_password",
        "restart_store_device",
    ],
}

# -----------------------------------------------------------------------------
# Mock data
# -----------------------------------------------------------------------------
MOCK_KB = [
    {
        "title": "POS terminal freezes on checkout",
        "tags": ["pos", "checkout", "freeze", "terminal"],
        "content": (
            "If a POS terminal freezes at checkout: (1) ask the associate to hold "
            "the power button 10s to force-restart, (2) confirm Wi-Fi signal on the "
            "terminal, (3) if it recurs more than twice in a shift, file a ticket "
            "with priority High and tag 'pos-hardware'."
        ),
    },
    {
        "title": "Pharmacy label printer not printing",
        "tags": ["printer", "pharmacy", "label", "rx"],
        "content": (
            "Pharmacy label printers (Zebra ZD-series) most often fail due to a "
            "roll misfeed. Reseat the label roll, close the lid firmly, and send a "
            "test label from the dispensing system. If the print queue shows a "
            "stuck job, clear it before retrying."
        ),
    },
    {
        "title": "Scan & Go app login errors",
        "tags": ["scan and go", "app", "login", "mobile"],
        "content": (
            "Scan & Go login failures are usually an expired store Wi-Fi "
            "certificate. Confirm the store's guest/associate network is online "
            "via check_system_status, then have the customer force-close and "
            "reopen the app."
        ),
    },
    {
        "title": "Password reset policy",
        "tags": ["password", "reset", "account", "login"],
        "content": (
            "Password resets for store-system accounts must be approved by an IT "
            "Admin before being executed, logged with the requester's name, and "
            "the user must change the temporary password on first login."
        ),
    },
    {
        "title": "Inventory sync failures",
        "tags": ["inventory", "sync", "stock"],
        "content": (
            "If a store's inventory sync is more than 6 hours stale, check the "
            "store's network status first. If network is healthy, escalate to "
            "on-call — stale syncs can cause incorrect out-of-stock displays."
        ),
    },
]

MOCK_SYSTEMS = {
    "POS-1042": {"name": "POS Terminal (Store #1042)", "status": "degraded", "detail": "Intermittent freeze on checkout screen"},
    "PHARM-PRINT-07": {"name": "Pharmacy Label Printer #07", "status": "offline", "detail": "No response — possible roll misfeed"},
    "WIFI-STORE-1042": {"name": "Store #1042 Guest/Associate Wi-Fi", "status": "healthy", "detail": "All access points online"},
    "SCANGO-APP": {"name": "Scan & Go Mobile App (backend)", "status": "healthy", "detail": "No active incidents"},
    "RX-DISPENSE-SYS": {"name": "Rx Dispensing System", "status": "healthy", "detail": "All automation cells reporting normal"},
}

MOCK_INVENTORY_SYNC = {
    "1042": {"last_sync": datetime.now() - timedelta(hours=7, minutes=12), "status": "stale"},
    "2201": {"last_sync": datetime.now() - timedelta(minutes=18), "status": "current"},
    "0588": {"last_sync": datetime.now() - timedelta(hours=1, minutes=5), "status": "current"},
}

# In-memory ticket counter for the demo session
_TICKET_COUNTER = {"n": 4812}


# -----------------------------------------------------------------------------
# LOW RISK — read-only tools
# -----------------------------------------------------------------------------
@tool
def search_knowledge_base(query: str) -> str:
    """Search internal IT knowledge base / runbooks for troubleshooting guidance.
    Use this first before filing a ticket or escalating."""
    q = query.lower()
    hits = [a for a in MOCK_KB if any(tag in q for tag in a["tags"]) or q in a["title"].lower()]
    if not hits:
        return "No knowledge base articles matched that query. Consider filing a ticket."
    out = []
    for a in hits[:3]:
        out.append(f"### {a['title']}\n{a['content']}")
    return "\n\n".join(out)


@tool
def check_system_status(system_id: str) -> str:
    """Check the live status of a store system by its ID, e.g. 'POS-1042',
    'PHARM-PRINT-07', 'WIFI-STORE-1042', 'SCANGO-APP', 'RX-DISPENSE-SYS'."""
    sys = MOCK_SYSTEMS.get(system_id.upper())
    if not sys:
        known = ", ".join(MOCK_SYSTEMS.keys())
        return f"Unknown system ID '{system_id}'. Known systems: {known}"
    return f"{sys['name']}: status={sys['status']} — {sys['detail']}"


@tool
def lookup_store_inventory_sync_status(store_id: str) -> str:
    """Look up when a store's inventory last successfully synced, by store ID,
    e.g. '1042'."""
    rec = MOCK_INVENTORY_SYNC.get(store_id)
    if not rec:
        return f"No inventory sync record found for store {store_id}."
    age = datetime.now() - rec["last_sync"]
    hours = round(age.total_seconds() / 3600, 1)
    return f"Store {store_id}: last sync {hours}h ago — status: {rec['status']}"


# -----------------------------------------------------------------------------
# MEDIUM RISK — write actions, but non-destructive and reversible
# -----------------------------------------------------------------------------
@tool
def create_itsm_ticket(summary: str, priority: str = "Medium") -> str:
    """File an IT support ticket. priority should be one of Low, Medium, High."""
    _TICKET_COUNTER["n"] += 1
    ticket_id = f"IT-{_TICKET_COUNTER['n']}"
    return f"Ticket {ticket_id} created (priority={priority}): {summary}"


@tool
def escalate_to_oncall(issue_summary: str) -> str:
    """Page the on-call engineer for an urgent issue that needs human attention now."""
    ref = str(uuid.uuid4())[:8]
    return f"🔔 On-call engineer paged (ref: {ref}) for: {issue_summary}"


# -----------------------------------------------------------------------------
# HIGH RISK — destructive / sensitive actions.
#
# IMPORTANT: these functions are the REAL actions, only ever invoked directly
# by the Streamlit approval-handling code AFTER a human clicks Approve — never
# bound directly to the agent. See agent.py's `make_guarded_tool` wrapper,
# which is what the agent actually calls, and which only ever QUEUES a
# request instead of running these.
# -----------------------------------------------------------------------------
def execute_reset_user_password(username: str) -> str:
    temp_pw = f"Temp{random.randint(1000, 9999)}!"
    return f"Password reset for '{username}'. Temporary password issued: {temp_pw} (must change on next login)."


def execute_restart_store_device(device_id: str) -> str:
    return f"Restart command sent to device '{device_id}'. Expected back online in ~2 minutes."


# Human-readable descriptions used in the approval UI
HIGH_RISK_DESCRIPTIONS = {
    "reset_user_password": "Reset a store employee's system password (issues a temporary password).",
    "restart_store_device": "Remotely restart a store device (POS terminal, printer, etc.) — interrupts active use.",
}

ALL_TOOLS = [
    search_knowledge_base,
    check_system_status,
    lookup_store_inventory_sync_status,
    create_itsm_ticket,
    escalate_to_oncall,
]
