# =============================================================================
# Pharmacy IT Operations Agent — Trace recorder + audit callback handler
# AI Friday
#
# This module is the single source of truth for "what happened, in what
# order" during one turn. Every stage — rate limiter, guardrails, retrieval,
# LLM calls, tool calls, retries, approvals, citation check — goes through
# Recorder.emit(), which fans out to:
#   1. the turn's own `steps` list  — shown live in the sidebar pipeline and
#                                     in the per-message "execution trace"
#   2. the SQL audit_log table      — the durable audit trail (Audit tab)
#
# Why a Recorder object instead of writing to st.session_state directly:
# LangGraph runs tools on worker threads, and Streamlit's session state is
# only safe to touch from the script thread. The Recorder is a plain Python
# object; the app copies its steps into session state after the turn ends.
# =============================================================================

import threading
import time
from datetime import datetime

from langchain_core.callbacks import BaseCallbackHandler

from agent import db

# Shared status vocabulary used everywhere a step is rendered.
STATUS_ICON = {
    "ok": "✅",
    "running": "⏳",
    "blocked": "🛑",
    "pending": "⏸️",
    "redacted": "🛡️",
    "error": "⚠️",
    "retry": "🔁",
}


def log_event(*, session_id: str, role: str, event_type: str, name: str, risk_tier: str = "n/a",
              input_data: str = "", output_data: str = "", status: str = "ok"):
    """Appends one row to the durable audit trail (SQL audit_log table)."""
    db.write_audit({
        "timestamp": db.now_str(),
        "session_id": session_id,
        "role": role,
        "event_type": event_type,
        "name": name,
        "risk_tier": risk_tier,
        "input": str(input_data)[:300],
        "output": str(output_data)[:300],
        "status": status,
    })


class Recorder:
    """Collects the ordered steps of ONE turn. Thread-safe."""

    def __init__(self, turn_id: str, session_id: str, role: str, on_step=None):
        self.turn_id = turn_id
        self.session_id = session_id
        self.role = role
        self.steps = []
        self._lock = threading.Lock()
        self._on_step = on_step
        self._main_thread = threading.current_thread()

    def emit(self, *, step_type: str, label: str, status: str = "ok", detail: str = "",
             latency_ms: float = None, risk_tier: str = "n/a", log_to_audit: bool = True,
             input_data: str = ""):
        with self._lock:
            step = {
                "step_no": len(self.steps) + 1,
                "turn_id": self.turn_id,
                "type": step_type,
                "label": label,
                "status": status,
                "detail": detail,
                "latency_ms": latency_ms,
                "time": datetime.now().strftime("%H:%M:%S"),
            }
            self.steps.append(step)

        if log_to_audit:
            log_event(session_id=self.session_id, role=self.role, event_type=step_type, name=label,
                      risk_tier=risk_tier, input_data=input_data, output_data=detail, status=status)

        # Live UI updates only from the Streamlit script thread; worker-thread
        # steps appear on the next render.
        if self._on_step and threading.current_thread() is self._main_thread:
            self._on_step(list(self.steps))
        return step


# Tools emit their own fine-grained steps (each retry attempt, each auth check)
# through whichever Recorder is active for the current turn.
_ACTIVE = {"recorder": None}


def set_active_recorder(recorder):
    _ACTIVE["recorder"] = recorder


def tool_emit(**kwargs):
    rec = _ACTIVE["recorder"]
    if rec is not None:
        rec.emit(**kwargs)


class AuditCallbackHandler(BaseCallbackHandler):
    """Hooks into LangChain's tool + LLM lifecycle and turns every call into
    a Recorder step, so the sidebar, trace and audit log all show the agent's
    reasoning and tool use in the exact order it happened."""

    def __init__(self, recorder: Recorder, tool_risk: dict):
        self.rec = recorder
        self.tool_risk = tool_risk
        self._tool_start = {}
        self._llm_start = {}
        self._llm_call_no = 0

    # ---- Tool lifecycle --------------------------------------------------
    def on_tool_start(self, serialized, input_str, *, run_id, **kwargs):
        name = (serialized or {}).get("name", "unknown_tool")
        self._tool_start[run_id] = (time.time(), name, input_str)
        self.rec.emit(step_type="tool_call", label=f"Tool call — {name}", status="running",
                      detail=f"input: {input_str}", risk_tier=self.tool_risk.get(name, "n/a"),
                      log_to_audit=False)

    def on_tool_end(self, output, *, run_id, **kwargs):
        started, name, input_str = self._tool_start.pop(run_id, (time.time(), "unknown_tool", ""))
        content = getattr(output, "content", output)
        text = str(content)
        status = "error" if '"ok": false' in text.lower() else "ok"
        self.rec.emit(step_type="tool_call", label=f"Tool call — {name}", status=status,
                      detail=text[:240], latency_ms=round((time.time() - started) * 1000, 1),
                      risk_tier=self.tool_risk.get(name, "n/a"), input_data=input_str)

    def on_tool_error(self, error, *, run_id, **kwargs):
        _, name, input_str = self._tool_start.pop(run_id, (None, "unknown_tool", ""))
        self.rec.emit(step_type="tool_call", label=f"Tool call — {name}", status="error",
                      detail=str(error)[:240], risk_tier=self.tool_risk.get(name, "n/a"),
                      input_data=input_str)

    # ---- LLM lifecycle -----------------------------------------------------
    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        self._start_llm(run_id)

    def on_llm_start(self, serialized, prompts, *, run_id, **kwargs):
        self._start_llm(run_id)

    def _start_llm(self, run_id):
        self._llm_call_no += 1
        self._llm_start[run_id] = (time.time(), self._llm_call_no)
        self.rec.emit(step_type="llm_call", label=f"LLM call #{self._llm_call_no} — reasoning...",
                      status="running", log_to_audit=False)

    def on_llm_end(self, response, *, run_id, **kwargs):
        started, call_no = self._llm_start.pop(run_id, (time.time(), self._llm_call_no))
        tokens = None
        try:
            usage = (response.llm_output or {}).get("token_usage")
            tokens = usage.get("total_tokens") if usage else None
        except Exception:
            pass
        detail = f"tokens={tokens}" if tokens else "tokens=n/a"
        try:
            msg = response.generations[0][0].message
            if getattr(msg, "tool_calls", None):
                detail = f"selected tool(s): {', '.join(tc['name'] for tc in msg.tool_calls)} · {detail}"
            else:
                detail = f"produced final answer · {detail}"
        except Exception:
            pass
        self.rec.emit(step_type="llm_call", label=f"LLM call #{call_no} completed", status="ok",
                      detail=detail, latency_ms=round((time.time() - started) * 1000, 1))
