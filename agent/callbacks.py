# =============================================================================
# Pharmacy IT Operations Agent — Audit Callback Handler
# Author : Abhinav Rawat · AI Architect · AI Playground
#
# This module is the single source of truth for "what happened, in what
# order" during one turn. Every stage of the pipeline — rate limiter, input
# guardrail, role allowlist, LLM calls, tool calls, output guardrail — goes
# through emit(), which fans out to three places:
#   1. st.session_state["trace"]       — full history, grouped by turn_id,
#                                         shown in the Observability tab
#   2. the turn's own `steps` list      — shown live in the sidebar pipeline
#                                         and in the per-message "View
#                                         execution trace" expander
#   3. st.session_state["audit_log"] +  — the durable audit trail (Audit Log
#      data/audit_log.csv                 tab + CSV on disk)
#
# Keeping ONE function (emit) as the entry point means the UI's live pipeline,
# the per-turn trace, and the audit log can never drift out of sync.
# =============================================================================

import csv
import os
import time
from datetime import datetime

from langchain_core.callbacks import BaseCallbackHandler

AUDIT_LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "audit_log.csv")

AUDIT_FIELDS = ["timestamp", "session_id", "role", "event_type", "name", "risk_tier", "input", "output", "status"]

# Shared status vocabulary used everywhere a step is rendered (sidebar,
# per-turn expander, Observability tab).
STATUS_ICON = {
    "ok": "✅",
    "running": "⏳",
    "blocked": "🛑",
    "pending": "⏸️",
    "redacted": "🛡️",
    "error": "⚠️",
}


def _ensure_csv_header():
    if not os.path.exists(AUDIT_LOG_PATH):
        os.makedirs(os.path.dirname(AUDIT_LOG_PATH), exist_ok=True)
        with open(AUDIT_LOG_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=AUDIT_FIELDS)
            writer.writeheader()


def log_event(session_state, *, session_id: str, role: str, event_type: str, name: str,
              risk_tier: str = "n/a", input_data: str = "", output_data: str = "", status: str = "ok"):
    """Appends one row to the durable audit trail (session_state + CSV)."""
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


def emit(session_state, steps_list, *, turn_id: str, session_id: str, role: str,
         step_type: str, label: str, status: str = "ok", detail: str = "",
         latency_ms: float = None, risk_tier: str = "n/a", log_to_audit: bool = True,
         on_step=None):
    """The single entry point for recording one pipeline step.

    - Appends to `steps_list` (the CURRENT turn's ordered sequence — used by
      the live sidebar pipeline and the per-message trace expander).
    - Appends to session_state["trace"] (the full cross-turn history — used
      by the Observability tab, grouped by turn_id).
    - Optionally writes to the durable audit trail (log_event + CSV).
    - Optionally calls on_step(steps_list) so the caller can re-render a live
      UI placeholder immediately, mid-script, as the turn progresses.
    """
    step = {
        "step_no": len(steps_list) + 1,
        "turn_id": turn_id,
        "type": step_type,
        "label": label,
        "status": status,
        "detail": detail,
        "latency_ms": latency_ms,
        "time": datetime.now().strftime("%H:%M:%S"),
    }
    steps_list.append(step)
    session_state.setdefault("trace", []).append(step)

    if log_to_audit:
        log_event(
            session_state, session_id=session_id, role=role, event_type=step_type,
            name=label, risk_tier=risk_tier, input_data="", output_data=detail, status=status,
        )

    if on_step:
        on_step(steps_list)

    return step


class AuditCallbackHandler(BaseCallbackHandler):
    """Hooks into LangChain's tool + LLM lifecycle and turns every call into
    an `emit()` step, scoped to one turn_id, so the sidebar/trace/audit log
    all show the agent's reasoning and tool use in the exact order it
    happened."""

    def __init__(self, session_state, session_id: str, role: str, tool_risk: dict,
                 turn_id: str, steps: list, on_step=None):
        self.session_state = session_state
        self.session_id = session_id
        self.role = role
        self.tool_risk = tool_risk
        self.turn_id = turn_id
        self.steps = steps
        self.on_step = on_step
        self._tool_start_times = {}
        self._tool_names = {}
        self._tool_inputs = {}
        self._llm_start_times = {}
        self._llm_call_no = 0

    def _emit(self, **kwargs):
        return emit(
            self.session_state, self.steps,
            turn_id=self.turn_id, session_id=self.session_id, role=self.role,
            on_step=self.on_step, **kwargs,
        )

    # ---- Tool lifecycle --------------------------------------------------
    def on_tool_start(self, serialized, input_str, *, run_id, **kwargs):
        name = (serialized or {}).get("name", "unknown_tool")
        self._tool_start_times[run_id] = time.time()
        self._tool_names[run_id] = name
        self._tool_inputs[run_id] = input_str
        self._emit(
            step_type="tool_call", label=f"Tool call — {name}", status="running",
            detail=f"input: {input_str}", risk_tier=self.tool_risk.get(name, "n/a"),
            log_to_audit=False,
        )

    def on_tool_end(self, output, *, run_id, **kwargs):
        name = self._tool_names.pop(run_id, "unknown_tool")
        started = self._tool_start_times.pop(run_id, time.time())
        self._tool_inputs.pop(run_id, "")
        latency_ms = round((time.time() - started) * 1000, 1)
        risk = self.tool_risk.get(name, "n/a")

        # High-risk tools only ever queue an approval — reflect that in the
        # step status rather than claiming the action completed.
        status = "pending" if risk == "high" else "ok"

        self._emit(
            step_type="tool_call", label=f"Tool call — {name}", status=status,
            detail=str(output)[:200], latency_ms=latency_ms, risk_tier=risk,
        )

    def on_tool_error(self, error, *, run_id, **kwargs):
        name = self._tool_names.pop(run_id, "unknown_tool")
        self._tool_start_times.pop(run_id, None)
        self._tool_inputs.pop(run_id, None)
        self._emit(
            step_type="tool_call", label=f"Tool call — {name}", status="error",
            detail=str(error), risk_tier=self.tool_risk.get(name, "n/a"),
        )

    # ---- LLM lifecycle (for latency / token metrics) ----------------------
    def on_llm_start(self, serialized, prompts, *, run_id, **kwargs):
        self._llm_call_no += 1
        self._llm_start_times[run_id] = (time.time(), self._llm_call_no)
        self._emit(
            step_type="llm_call", label=f"LLM call #{self._llm_call_no} — reasoning...",
            status="running", log_to_audit=False,
        )

    def on_llm_end(self, response, *, run_id, **kwargs):
        started, call_no = self._llm_start_times.pop(run_id, (time.time(), self._llm_call_no))
        latency_ms = round((time.time() - started) * 1000, 1)

        tokens = None
        try:
            usage = response.llm_output.get("token_usage") if response.llm_output else None
            if usage:
                tokens = usage.get("total_tokens")
        except Exception:
            tokens = None

        # Peek at whether this LLM turn produced a tool call or a final answer.
        detail = f"tokens={tokens}" if tokens else "tokens=n/a"
        try:
            gen = response.generations[0][0]
            tool_calls = getattr(gen.message, "tool_calls", None) if hasattr(gen, "message") else None
            if tool_calls:
                names = ", ".join(tc.get("name", "?") for tc in tool_calls)
                detail = f"selected tool(s): {names} · {detail}"
            else:
                detail = f"produced final answer · {detail}"
        except Exception:
            pass

        self._emit(
            step_type="llm_call", label=f"LLM call #{call_no} completed",
            status="ok", detail=detail, latency_ms=latency_ms,
        )
