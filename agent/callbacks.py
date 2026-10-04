# =============================================================================
# Pharmacy IT Operations Agent — Audit Callback Handler
# Author : Abhinav Rawat · AI Architect · AI Playground
#
# A LangChain BaseCallbackHandler that captures every tool call (and LLM call,
# for token/latency metrics) into:
#   1. st.session_state["audit_log"]  — shown live in the Audit Log tab
#   2. st.session_state["trace"]      — shown live in the Observability tab
#   3. data/audit_log.csv             — persisted across app restarts
#
# This is deliberately separate from guardrails.py / rate_limiter.py: audit
# logging should capture EVERYTHING (including blocked/denied actions), not
# just successful tool calls.
# =============================================================================

import csv
import os
import time
import uuid
from datetime import datetime

from langchain_core.callbacks import BaseCallbackHandler

AUDIT_LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "audit_log.csv")

AUDIT_FIELDS = ["timestamp", "session_id", "role", "event_type", "name", "risk_tier", "input", "output", "status"]


def _ensure_csv_header():
    if not os.path.exists(AUDIT_LOG_PATH):
        os.makedirs(os.path.dirname(AUDIT_LOG_PATH), exist_ok=True)
        with open(AUDIT_LOG_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=AUDIT_FIELDS)
            writer.writeheader()


def log_event(session_state, *, session_id: str, role: str, event_type: str, name: str,
              risk_tier: str = "n/a", input_data: str = "", output_data: str = "", status: str = "ok"):
    """Central logging function used for both tool calls AND non-tool events
    (guardrail blocks, rate-limit hits, approvals, denials)."""
    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "session_id": session_id,
        "role": role,
        "event_type": event_type,
        "name": name,
        "risk_tier": risk_tier,
        "input": str(input_data)[:300],
        "output": str(output_data)[:300],
        "status": status,
    }

    session_state.setdefault("audit_log", [])
    session_state["audit_log"].append(record)

    _ensure_csv_header()
    with open(AUDIT_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=AUDIT_FIELDS)
        writer.writerow(record)


class AuditCallbackHandler(BaseCallbackHandler):
    """Hooks into LangChain's tool + LLM lifecycle to populate the audit log
    and the observability trace automatically for every agent run."""

    def __init__(self, session_state, session_id: str, role: str, tool_risk: dict):
        self.session_state = session_state
        self.session_id = session_id
        self.role = role
        self.tool_risk = tool_risk
        self._tool_start_times = {}
        self._llm_start_times = {}
        self.session_state.setdefault("trace", [])

    # ---- Tool lifecycle --------------------------------------------------
    def on_tool_start(self, serialized, input_str, *, run_id, **kwargs):
        self._tool_start_times[run_id] = time.time()

    def on_tool_end(self, output, *, run_id, **kwargs):
        name = kwargs.get("name") or "unknown_tool"
        started = self._tool_start_times.pop(run_id, time.time())
        latency_ms = round((time.time() - started) * 1000, 1)
        risk = self.tool_risk.get(name, "n/a")

        log_event(
            self.session_state,
            session_id=self.session_id,
            role=self.role,
            event_type="tool_call",
            name=name,
            risk_tier=risk,
            input_data=kwargs.get("inputs", ""),
            output_data=output,
            status="ok",
        )
        self.session_state["trace"].append({
            "id": str(uuid.uuid4())[:8],
            "type": "tool_call",
            "name": name,
            "risk_tier": risk,
            "latency_ms": latency_ms,
            "output_preview": str(output)[:160],
        })

    def on_tool_error(self, error, *, run_id, **kwargs):
        name = kwargs.get("name") or "unknown_tool"
        log_event(
            self.session_state,
            session_id=self.session_id,
            role=self.role,
            event_type="tool_call",
            name=name,
            risk_tier=self.tool_risk.get(name, "n/a"),
            input_data="",
            output_data=str(error),
            status="error",
        )

    # ---- LLM lifecycle (for latency / token metrics) ----------------------
    def on_llm_start(self, serialized, prompts, *, run_id, **kwargs):
        self._llm_start_times[run_id] = time.time()

    def on_llm_end(self, response, *, run_id, **kwargs):
        started = self._llm_start_times.pop(run_id, time.time())
        latency_ms = round((time.time() - started) * 1000, 1)

        tokens = None
        try:
            usage = response.llm_output.get("token_usage") if response.llm_output else None
            if usage:
                tokens = usage.get("total_tokens")
        except Exception:
            tokens = None

        self.session_state["trace"].append({
            "id": str(uuid.uuid4())[:8],
            "type": "llm_call",
            "name": "chat_model",
            "risk_tier": "n/a",
            "latency_ms": latency_ms,
            "output_preview": f"tokens={tokens}" if tokens else "tokens=n/a",
        })
