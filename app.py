# =============================================================================
# Pharmacy IT Operations Agent — Streamlit Demo
# Author : Abhinav Rawat · AI Architect · AI Playground
#
# A companion training demo to the RAG AI Assistant. Where that demo teaches
# retrieval, this one teaches: tool-calling agents + guardrails +
# observability + audit logging + rate limiting + human-in-the-loop approval.
#
# Same visual theme as the RAG demo (dark navy / teal / purple, Space Mono +
# DM Sans) so the two feel like one training series.
# =============================================================================

import os
import uuid

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

from agent.agent import build_agent
from agent.callbacks import AuditCallbackHandler, log_event, emit, STATUS_ICON, AUDIT_LOG_PATH
from agent.guardrails import check_prompt_injection, filter_output_pii
from agent.rate_limiter import check_and_consume, current_usage
from agent.tools import (
    TOOL_RISK,
    ROLE_TOOL_ACCESS,
    HIGH_RISK_DESCRIPTIONS,
    execute_reset_user_password,
    execute_restart_store_device,
)

load_dotenv()

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Pharmacy IT Operations Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# CSS — same theme family as the RAG demo, plus agent-specific badges
# =============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@400;500;600&display=swap');

:root {
    --bg:      #080C14;
    --surface: #0D1320;
    --card:    #131C2E;
    --border:  #253450;
    --accent:  #00D4AA;
    --purple:  #8B7FFF;
    --amber:   #F5A623;
    --green:   #4ADE80;
    --red:     #FF6B6B;
    --text:    #F0F2FF;
    --sub:     #A8B4CC;
    --muted:   #6A7A96;
    --mono:    'Space Mono', monospace;
    --sans:    'DM Sans', sans-serif;
}

html, body, [class*="css"] {
    font-family: var(--sans) !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}
.stApp { background-color: var(--bg) !important; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem !important; max-width: 1200px !important; }

section[data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] .block-container { padding: 1.5rem 1.25rem !important; }

label, .stTextInput label, .stTextArea label, .stRadio label,
[data-testid="stWidgetLabel"] p {
    color: var(--sub) !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    font-family: var(--sans) !important;
    margin-bottom: 6px !important;
}

.stTextInput > div > div > input, .stTextArea > div > div > textarea {
    background: var(--card) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-family: var(--sans) !important;
    font-size: 15px !important;
    padding: 12px 16px !important;
}
.stTextInput > div > div > input:focus { border-color: var(--accent) !important; }
input[type="password"] { font-family: var(--mono) !important; letter-spacing: 3px !important; }

.stButton > button {
    background: linear-gradient(135deg, #00D4AA, #6C63FF) !important;
    color: #080C14 !important;
    font-family: var(--mono) !important;
    font-size: 13px !important;
    font-weight: 700 !important;
    letter-spacing: 0.5px !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 20px !important;
    width: 100% !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

button[kind="secondary"], .deny-btn button {
    background: linear-gradient(135deg, #FF6B6B, #C94444) !important;
}

.stTabs [data-baseweb="tab-list"] {
    background: var(--surface) !important;
    border-radius: 10px !important;
    padding: 4px !important;
    gap: 4px !important;
    border: 1px solid var(--border) !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 8px !important;
    color: var(--muted) !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    padding: 8px 16px !important;
    border: none !important;
}
.stTabs [aria-selected="true"] {
    background: var(--card) !important;
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
}
.stTabs [data-baseweb="tab-panel"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 0 0 12px 12px !important;
    padding: 1.25rem !important;
}

.section-label {
    font-family: var(--mono); font-size: 11px; letter-spacing: 2.5px;
    color: var(--accent); text-transform: uppercase;
    margin-bottom: 0.5rem; margin-top: 1rem; display: block;
}
.sidebar-header {
    font-family: var(--mono); font-size: 11px; letter-spacing: 2.5px;
    color: var(--accent); text-transform: uppercase;
    padding-bottom: 10px; border-bottom: 1px solid var(--border); margin-bottom: 1rem;
}
.custom-divider {
    border: none; height: 1px;
    background: linear-gradient(90deg, transparent, var(--border), transparent);
    margin: 1.5rem 0;
}
.hero-title { font-family: var(--mono); font-size: 2.4rem; font-weight: 700; color: var(--accent); margin: 0 0 0.5rem; line-height: 1.1; }
.hero-sub { font-size: 16px; color: var(--sub); font-weight: 400; margin: 0 0 1.5rem; }

.stats-row { display: flex; gap: 1rem; margin: 1.2rem 0; flex-wrap: wrap; }
.stat-card { flex: 1; min-width: 120px; background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 1rem; text-align: center; }
.stat-value { font-family: var(--mono); font-size: 1.6rem; font-weight: 700; color: var(--accent); }
.stat-label { font-size: 12px; color: var(--muted); margin-top: 4px; }

[data-testid="stInfo"]    { background:#253450!important;border:1px solid #3A5080!important;color:var(--sub)!important; }
[data-testid="stSuccess"] { background:#0D2A1E!important;border:1px solid #1A5C38!important;color:#4ADE80!important; }
[data-testid="stWarning"] { background:#2A1E00!important;border:1px solid #5C3D00!important;color:#F5A623!important; }
[data-testid="stError"]   { background:#2A0D0D!important;border:1px solid #5C1A1A!important;color:#FF6B6B!important; }

/* ── Badges ── */
.badge { display:inline-block; font-family: var(--mono); font-size: 11px; letter-spacing: 1px;
         padding: 4px 10px; border-radius: 20px; margin: 2px 6px 2px 0; text-transform: uppercase; }
.badge-low      { background:#0D2A1E; border:1px solid #1A5C38; color:#4ADE80; }
.badge-medium   { background:#2A1E00; border:1px solid #5C3D00; color:#F5A623; }
.badge-high     { background:#2A0D0D; border:1px solid #5C1A1A; color:#FF6B6B; }
.badge-blocked  { background:#2A0D0D; border:1px solid #5C1A1A; color:#FF6B6B; }
.badge-pending  { background:#2A1E00; border:1px solid #5C3D00; color:#F5A623; }
.badge-approved { background:#0D2A1E; border:1px solid #1A5C38; color:#4ADE80; }
.badge-denied   { background:#2A0D0D; border:1px solid #5C1A1A; color:#FF6B6B; }

/* ── Live pipeline (sidebar + per-turn trace) ── */
.pipeline-box { display: flex; flex-direction: column; gap: 6px; margin-top: 4px; }
.pipeline-step {
    background: var(--card); border: 1px solid var(--border); border-left: 3px solid var(--muted);
    border-radius: 8px; padding: 6px 10px;
}
.pipeline-step.step-ok        { border-left-color: var(--green); }
.pipeline-step.step-running   { border-left-color: var(--purple); }
.pipeline-step.step-blocked   { border-left-color: var(--red); }
.pipeline-step.step-pending   { border-left-color: var(--amber); }
.pipeline-step.step-error     { border-left-color: var(--red); }
.pipeline-step.step-redacted  { border-left-color: var(--amber); }
.pipeline-icon { margin-right: 6px; }
.pipeline-label {
    font-family: var(--mono); font-size: 12px; font-weight: 700; color: var(--text);
}
.pipeline-detail {
    font-size: 11.5px; color: var(--muted); margin-top: 2px; padding-left: 20px;
    word-break: break-word;
}


.answer-card {
    background: var(--card); border: 1px solid var(--border); border-left: 3px solid var(--accent);
    border-radius: 14px; padding: 1.25rem 1.5rem; margin: 0.75rem 0; line-height: 1.75; font-size: 15px; color: var(--text);
}
.source-card {
    background: var(--surface); border: 1px solid var(--border); border-left: 3px solid var(--purple);
    border-radius: 10px; padding: 0.9rem 1.1rem; margin: 6px 0; font-size: 14px; color: var(--sub); line-height: 1.7;
}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SESSION STATE
# =============================================================================
for key, default in {
    "api_key": os.getenv("OPENAI_API_KEY", ""),
    "role": "Store Associate",
    "session_id": str(uuid.uuid4())[:8],
    "chat_history": [],       # list of {"role": "user"/"assistant", "content": str, "badges": [...]}
    "pending_approvals": {},  # request_id -> {tool_name, arg_name, arg_value, status}
    "audit_log": [],
    "trace": [],
    "rate_limit_timestamps": [],
    "last_turn_steps": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# =============================================================================
# LIVE PIPELINE RENDERER
# Renders the ordered sequence of steps for one turn as a vertical list of
# cards. Called repeatedly, mid-script, as each stage of a turn completes —
# both to populate the sidebar's "Live Pipeline" placeholder in real time,
# and to render the frozen per-message trace in the Chat tab.
# =============================================================================
STATUS_CLASS = {
    "ok": "step-ok", "running": "step-running", "blocked": "step-blocked",
    "pending": "step-pending", "error": "step-error", "redacted": "step-redacted",
}


def render_pipeline(container, steps):
    if not steps:
        container.markdown(
            '<p style="font-size:12px;color:#6A7A96;">No activity yet this turn.</p>',
            unsafe_allow_html=True,
        )
        return
    rows = []
    for s in steps:
        icon = STATUS_ICON.get(s["status"], "•")
        cls = STATUS_CLASS.get(s["status"], "step-ok")
        lat = f' · {s["latency_ms"]}ms' if s.get("latency_ms") else ""
        detail = s.get("detail") or ""
        rows.append(
            f'<div class="pipeline-step {cls}">'
            f'<span class="pipeline-label"><span class="pipeline-icon">{icon}</span>'
            f'{s["step_no"]}. {s["label"]}</span>'
            f'<div class="pipeline-detail">{detail}{lat}</div>'
            f'</div>'
        )
    container.markdown('<div class="pipeline-box">' + "".join(rows) + '</div>', unsafe_allow_html=True)


# =============================================================================
# HERO SECTION
# =============================================================================
st.markdown("""
<div style="text-align:center; padding:2.5rem 1rem 1.5rem;">
    <div style="display:inline-block; background:#0D1320; border:1px solid #253450;
                border-radius:30px; padding:6px 20px; margin-bottom:1.2rem;">
        <span style="font-family:'Space Mono',monospace; font-size:11px;
                     letter-spacing:3px; color:#00D4AA; text-transform:uppercase;">
            🤖 Agentic AI · Guardrails · Observability
        </span>
    </div>
    <div class="hero-title">Pharmacy IT Operations Agent</div>
    <div class="hero-sub">Check systems, file tickets, or request access changes — watch every guardrail fire in real time</div>
    <div style="display:inline-block; background:#0D1320; border:1px solid #253450;
                border-radius:40px; padding:10px 28px;">
        <span style="font-family:'Space Mono',monospace; font-size:12px; color:#6A7A96; letter-spacing:1px;">Built by &nbsp;</span>
        <span style="font-family:'Space Mono',monospace; font-size:14px; font-weight:700; color:#00D4AA; letter-spacing:1px;">Abhinav Rawat</span>
        <span style="font-family:'Space Mono',monospace; font-size:12px; color:#6A7A96; letter-spacing:1px;">&nbsp;·&nbsp;AI Playground</span>
    </div>
</div>
""", unsafe_allow_html=True)
st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown('<div class="sidebar-header">⚙ Configuration</div>', unsafe_allow_html=True)

    st.markdown('<span class="section-label">🔑 OpenAI API Key</span>', unsafe_allow_html=True)
    api_input = st.text_input(
        label="OpenAI API Key", label_visibility="collapsed", type="password",
        value=st.session_state.api_key, placeholder="sk-proj-...", key="api_key_input",
    )
    if api_input:
        st.session_state.api_key = api_input
    active_key = st.session_state.api_key

    st.markdown('<hr style="border-color:#253450;margin:1.2rem 0;">', unsafe_allow_html=True)

    st.markdown('<span class="section-label">👤 Role</span>', unsafe_allow_html=True)
    role = st.radio(
        "Role", options=["Store Associate", "IT Admin"],
        index=["Store Associate", "IT Admin"].index(st.session_state.role),
        label_visibility="collapsed", key="role_radio",
    )
    st.session_state.role = role

    allowed = ROLE_TOOL_ACCESS[role]
    st.markdown(
        f'<p style="font-size:12px;color:#6A7A96;margin-top:6px;">'
        f'Tools available: {", ".join(allowed)}</p>',
        unsafe_allow_html=True,
    )

    st.markdown('<hr style="border-color:#253450;margin:1.2rem 0;">', unsafe_allow_html=True)

    st.markdown('<span class="section-label">⏱ Rate Limit</span>', unsafe_allow_html=True)
    used, limit = current_usage(st.session_state)
    st.progress(min(used / limit, 1.0))
    st.markdown(
        f'<p style="font-size:12px;color:#6A7A96;">{used}/{limit} requests used in the last 60s</p>',
        unsafe_allow_html=True,
    )

    st.markdown('<hr style="border-color:#253450;margin:1.2rem 0;">', unsafe_allow_html=True)

    st.markdown('<span class="section-label">🧭 Live Pipeline (this turn)</span>', unsafe_allow_html=True)
    pipeline_box = st.empty()
    render_pipeline(pipeline_box, st.session_state.last_turn_steps)

    st.markdown('<hr style="border-color:#253450;margin:1.2rem 0;">', unsafe_allow_html=True)

    if st.button("🔄 Reset session"):
        for key in ["chat_history", "pending_approvals", "audit_log", "trace", "rate_limit_timestamps", "last_turn_steps"]:
            st.session_state[key] = [] if isinstance(st.session_state[key], list) else {}
        st.session_state.session_id = str(uuid.uuid4())[:8]
        st.rerun()

    st.markdown('<hr style="border-color:#253450;margin:1.2rem 0;">', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:13px;color:#6A7A96;line-height:1.9;">
        <b style="color:#A8B4CC;font-size:14px;">Pipeline stages:</b><br>
        ⏱ Rate limiter → 🛡 Input guardrail → 🔐 Role allowlist →<br>
        🧠 LLM calls → 🔧 Tool calls → ⏸ Approval (high-risk) →<br>
        🛡 Output guardrail → 💬 Response<br><br>
        Every stage above shows live, in order, once you send a message.
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# HELPERS
# =============================================================================

def badge(text, variant):
    return f'<span class="badge badge-{variant}">{text}</span>'


def render_pending_approvals():
    pending = {rid: req for rid, req in st.session_state.pending_approvals.items() if req["status"] == "pending"}
    if not pending:
        return
    st.markdown('<span class="section-label">⏸ Awaiting Your Approval</span>', unsafe_allow_html=True)
    for rid, req in pending.items():
        desc = HIGH_RISK_DESCRIPTIONS.get(req["tool_name"], req["tool_name"])
        st.warning(
            f"**{req['tool_name']}** on `{req['arg_value']}` — {desc}\n\nRequest ID: `{rid}`"
        )
        col1, col2, _ = st.columns([1, 1, 3])
        with col1:
            if st.button("✅ Approve", key=f"approve_{rid}"):
                _resolve_approval(rid, approved=True)
                st.rerun()
        with col2:
            if st.button("❌ Deny", key=f"deny_{rid}"):
                _resolve_approval(rid, approved=False)
                st.rerun()


def _resolve_approval(request_id, approved: bool):
    req = st.session_state.pending_approvals[request_id]
    tool_name, arg_value = req["tool_name"], req["arg_value"]

    if approved:
        if tool_name == "reset_user_password":
            result = execute_reset_user_password(arg_value)
        elif tool_name == "restart_store_device":
            result = execute_restart_store_device(arg_value)
        else:
            result = "Unknown high-risk tool."
        req["status"] = "approved"
        note = f"✅ **Approved** by {st.session_state.role}: {result}"
        status = "approved"
    else:
        req["status"] = "denied"
        note = f"❌ **Denied** by {st.session_state.role}: `{tool_name}` on `{arg_value}` was not executed."
        status = "denied"

    log_event(
        st.session_state, session_id=st.session_state.session_id, role=st.session_state.role,
        event_type="approval", name=tool_name, risk_tier=TOOL_RISK.get(tool_name, "high"),
        input_data=arg_value, output_data=note, status=status,
    )
    st.session_state.chat_history.append({"role": "assistant", "content": note, "badges": [("APPROVAL", status)]})


def history_to_messages(history):
    messages = []
    for turn in history[:-1] if history and history[-1]["role"] == "user" else history:
        if turn["role"] == "user":
            messages.append(HumanMessage(content=turn["content"]))
        else:
            messages.append(AIMessage(content=turn["content"]))
    return messages


# =============================================================================
# TABS
# =============================================================================
tab_chat, tab_obs, tab_audit, tab_about = st.tabs(
    ["💬 Chat", "📊 Observability", "📋 Audit Log", "📘 Guardrails & Do/Don'ts"]
)

# ---- CHAT TAB ---------------------------------------------------------------
with tab_chat:
    render_pending_approvals()
    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    for turn in st.session_state.chat_history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            for label, variant in turn.get("badges", []):
                st.markdown(badge(label, variant.lower()), unsafe_allow_html=True)
            steps = turn.get("steps")
            if steps:
                total_latency = sum(s["latency_ms"] for s in steps if s.get("latency_ms"))
                with st.expander(f"🔍 View execution trace ({len(steps)} steps, {round(total_latency)}ms)"):
                    for s in steps:
                        icon = STATUS_ICON.get(s["status"], "•")
                        lat = f" · {s['latency_ms']}ms" if s.get("latency_ms") else ""
                        st.markdown(
                            f"**{s['step_no']}. {icon} {s['label']}**{lat}  \n"
                            f"<span style='color:#6A7A96;font-size:12.5px;'>{s.get('detail', '')}</span>",
                            unsafe_allow_html=True,
                        )

    user_msg = st.chat_input("Ask the Pharmacy IT Operations Agent...")

    if user_msg:
        if not active_key:
            st.error("⚠️ Enter your OpenAI API key in the sidebar first.")
        else:
            st.session_state.chat_history.append({"role": "user", "content": user_msg, "badges": []})

            turn_id = f"turn_{uuid.uuid4().hex[:8]}"
            current_steps = []

            def _live_render(steps):
                render_pipeline(pipeline_box, steps)

            allowed_req, retry_after, remaining = check_and_consume(st.session_state)
            emit(
                st.session_state, current_steps, turn_id=turn_id,
                session_id=st.session_state.session_id, role=st.session_state.role,
                step_type="rate_limit", label="Rate Limiter",
                status="ok" if allowed_req else "blocked",
                detail=(f"{remaining} requests remaining this window" if allowed_req
                        else f"Limit reached — retry in {retry_after}s"),
                on_step=_live_render,
            )
            if not allowed_req:
                msg = f"⏳ Rate limit reached. Try again in {retry_after}s."
                st.session_state.chat_history.append({
                    "role": "assistant", "content": msg, "badges": [("RATE LIMITED", "blocked")],
                    "steps": current_steps, "turn_id": turn_id,
                })
                st.session_state.last_turn_steps = current_steps
                st.rerun()

            blocked, reason = check_prompt_injection(user_msg)
            emit(
                st.session_state, current_steps, turn_id=turn_id,
                session_id=st.session_state.session_id, role=st.session_state.role,
                step_type="input_guardrail", label="Input Guardrail",
                status="blocked" if blocked else "ok",
                detail=reason if blocked else "No injection pattern matched",
                on_step=_live_render,
            )
            if blocked:
                msg = f"🛡️ Blocked by input guardrail: {reason}"
                st.session_state.chat_history.append({
                    "role": "assistant", "content": msg, "badges": [("BLOCKED", "blocked")],
                    "steps": current_steps, "turn_id": turn_id,
                })
                st.session_state.last_turn_steps = current_steps
                st.rerun()

            with st.spinner("🤖 Thinking..."):
                try:
                    executor, tools = build_agent(active_key, st.session_state.role, st.session_state.pending_approvals)
                    emit(
                        st.session_state, current_steps, turn_id=turn_id,
                        session_id=st.session_state.session_id, role=st.session_state.role,
                        step_type="tool_allowlist", label="Role Allowlist",
                        detail=f"{len(tools)} tool(s) bound for '{st.session_state.role}': "
                               f"{', '.join(t.name for t in tools)}",
                        on_step=_live_render,
                    )

                    callback = AuditCallbackHandler(
                        st.session_state, st.session_state.session_id, st.session_state.role, TOOL_RISK,
                        turn_id, current_steps, on_step=_live_render,
                    )
                    chat_history_msgs = history_to_messages(st.session_state.chat_history)
                    result = executor.invoke(
                        {"input": user_msg, "chat_history": chat_history_msgs},
                        config={"callbacks": [callback]},
                    )
                    answer = result.get("output", "")

                    cleaned, redactions = filter_output_pii(answer)
                    badges = []
                    emit(
                        st.session_state, current_steps, turn_id=turn_id,
                        session_id=st.session_state.session_id, role=st.session_state.role,
                        step_type="output_guardrail", label="Output Guardrail",
                        status="redacted" if redactions else "ok",
                        detail=f"Redacted: {', '.join(redactions)}" if redactions else "No PII/secrets detected",
                        on_step=_live_render,
                    )
                    if redactions:
                        badges.append((f"REDACTED: {', '.join(redactions)}", "blocked"))

                    emit(
                        st.session_state, current_steps, turn_id=turn_id,
                        session_id=st.session_state.session_id, role=st.session_state.role,
                        step_type="response", label="Response Delivered", log_to_audit=False,
                        on_step=_live_render,
                    )

                    st.session_state.chat_history.append({
                        "role": "assistant", "content": cleaned, "badges": badges,
                        "steps": current_steps, "turn_id": turn_id,
                    })
                except Exception as e:
                    emit(
                        st.session_state, current_steps, turn_id=turn_id,
                        session_id=st.session_state.session_id, role=st.session_state.role,
                        step_type="error", label="Agent Error", status="error", detail=str(e),
                        on_step=_live_render,
                    )
                    st.session_state.chat_history.append({
                        "role": "assistant", "content": f"⚠️ Agent error: {e}",
                        "badges": [("ERROR", "blocked")], "steps": current_steps, "turn_id": turn_id,
                    })
            st.session_state.last_turn_steps = current_steps
            st.rerun()

# ---- OBSERVABILITY TAB -------------------------------------------------------
with tab_obs:
    st.markdown('<span class="section-label">📊 Live Trace — grouped by turn</span>', unsafe_allow_html=True)
    trace = st.session_state.trace
    if not trace:
        st.info("No activity yet — send a message in the Chat tab to generate a trace.")
    else:
        total_latency = sum(t["latency_ms"] for t in trace if t.get("latency_ms"))
        tool_calls = [t for t in trace if t["type"] == "tool_call" and t["status"] != "running"]
        llm_calls = [t for t in trace if t["type"] == "llm_call" and t["status"] != "running"]
        st.markdown(f"""
        <div class="stats-row">
            <div class="stat-card"><div class="stat-value">{len(trace)}</div><div class="stat-label">TOTAL STEPS</div></div>
            <div class="stat-card"><div class="stat-value">{len(tool_calls)}</div><div class="stat-label">TOOL CALLS</div></div>
            <div class="stat-card"><div class="stat-value">{len(llm_calls)}</div><div class="stat-label">LLM CALLS</div></div>
            <div class="stat-card"><div class="stat-value">{round(total_latency)}ms</div><div class="stat-label">TOTAL LATENCY</div></div>
        </div>
        """, unsafe_allow_html=True)

        # Group steps by turn_id, most recent turn first, preserving in-turn order.
        turn_order = []
        turns = {}
        for s in trace:
            tid = s.get("turn_id", "n/a")
            if tid not in turns:
                turns[tid] = []
                turn_order.append(tid)
            turns[tid].append(s)

        for tid in reversed(turn_order):
            steps = turns[tid]
            turn_latency = sum(s["latency_ms"] for s in steps if s.get("latency_ms"))
            with st.expander(f"🧵 {tid} — {len(steps)} steps · {round(turn_latency)}ms", expanded=(tid == turn_order[-1])):
                df = pd.DataFrame(steps)[["step_no", "time", "type", "label", "status", "latency_ms", "detail"]]
                st.dataframe(df, use_container_width=True, hide_index=True)


# ---- AUDIT LOG TAB ------------------------------------------------------------
with tab_audit:
    st.markdown('<span class="section-label">📋 Audit Trail</span>', unsafe_allow_html=True)
    log = st.session_state.audit_log
    if not log:
        st.info("No audit events yet — every tool call, guardrail block, rate-limit hit, and approval decision will appear here.")
    else:
        blocked = len([e for e in log if e["status"] in ("blocked",)])
        approved = len([e for e in log if e["status"] == "approved"])
        denied = len([e for e in log if e["status"] == "denied"])
        st.markdown(f"""
        <div class="stats-row">
            <div class="stat-card"><div class="stat-value">{len(log)}</div><div class="stat-label">TOTAL EVENTS</div></div>
            <div class="stat-card"><div class="stat-value">{blocked}</div><div class="stat-label">GUARDRAIL/RATE BLOCKS</div></div>
            <div class="stat-card"><div class="stat-value">{approved}</div><div class="stat-label">APPROVED ACTIONS</div></div>
            <div class="stat-card"><div class="stat-value">{denied}</div><div class="stat-label">DENIED ACTIONS</div></div>
        </div>
        """, unsafe_allow_html=True)

        df = pd.DataFrame(log[::-1])
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.download_button(
            "⬇ Export audit_log.csv",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="audit_log.csv",
            mime="text/csv",
        )
        st.caption(f"Also persisted continuously to: `{AUDIT_LOG_PATH}`")

# ---- ABOUT / GUARDRAILS TAB ----------------------------------------------------
with tab_about:
    st.markdown('<span class="section-label">📘 How This Demo Is Built</span>', unsafe_allow_html=True)
    st.markdown("""
    <div class="answer-card">
    <b>Architecture</b><br><br>
    <code>User → Rate Limiter → Input Guardrail → Role Tool-Allowlist → AgentExecutor
    (LangChain tool-calling agent) → [high-risk tool? → Approval Queue] →
    Output Guardrail → Response</code><br><br>
    Every step writes to the Audit Log, whether it succeeds, gets blocked, or
    needs human approval.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<span class="section-label">🎯 Tool Risk Tiers</span>', unsafe_allow_html=True)
    risk_rows = "".join(
        f"<tr><td style='padding:6px 12px;'>{name}</td>"
        f"<td style='padding:6px 12px;'>{badge(risk.upper(), risk)}</td></tr>"
        for name, risk in TOOL_RISK.items()
    )
    st.markdown(f"""
    <div class="source-card">
    <table style="width:100%; border-collapse:collapse;">
    <tr><th style="text-align:left;padding:6px 12px;color:#6A7A96;">Tool</th>
        <th style="text-align:left;padding:6px 12px;color:#6A7A96;">Risk Tier</th></tr>
    {risk_rows}
    </table>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<span class="section-label">✅ Do / ❌ Don\'t</span>', unsafe_allow_html=True)
    st.markdown("""
    <div class="source-card">
    ✅ <b>Do</b> make destructive tools structurally incapable of acting without approval — don't rely on prompt instructions alone.<br><br>
    ❌ <b>Don't</b> bind high-risk tools directly to the agent and trust the system prompt to gate them.<br><br>
    ✅ <b>Do</b> apply guardrails as explicit steps before AND after the LLM call.<br><br>
    ❌ <b>Don't</b> skip output filtering — models can echo back secrets or PII from context/tool results.<br><br>
    ✅ <b>Do</b> rate-limit at the tool/session layer, not just the chat box.<br><br>
    ❌ <b>Don't</b> log only successes — audit guardrail blocks, denials, and errors too.<br><br>
    ✅ <b>Do</b> scope tool access by role (allowlist), not just by prompt.<br><br>
    ❌ <b>Don't</b> assume a mocked demo's simplicity is "good enough" for production — treat this as a pattern to harden further (real auth, real approvals workflow, real moderation API).
    </div>
    """, unsafe_allow_html=True)
