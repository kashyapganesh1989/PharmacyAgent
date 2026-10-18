# 🤖 Pharmacy IT Operations Agent

A hands-on workshop app for **tool-calling agents**. The **🧭 Guided Demo**
tab walks through eight steps one at a time, and the **Next** button stays
locked until the current step's checkpoint passes:

| Step | What happens | Checkpoint before moving on |
|---|---|---|
| 0. Health check | API key, SQLite, LangGraph checkpointer, mock search | All checks green |
| 1. Upload Document | Type and size validated before storage | Document listed, upload audited |
| 2. Ingest to Vector DB | Extract → chunk → embed → FAISS on disk | Index count matches chunks in SQL |
| 3. Retrieve | Top-k passages with cosine scores; threshold shown | At least one passage above threshold |
| 4. Ground | Agent answers only from passages, cites chunk IDs; citations checked in code | Grounded answer + off-topic "not covered" |
| 5. External Search | Mock inventory search; then simulated timeout → 3 attempts → honest fallback | Success and fallback both seen |
| 6. Reset SQL | Dry run, then the graph **pauses** until an IT Admin approves | Reset ran only after approval, audited |
| 7. Update User Role | Store Associate refused; admin change approved; reset link to mock mailbox | Token never appears in chat or trace |
| 8. Restart App | Services rebuilt; conversation restored from the SQLite checkpoint | Same message count before and after |

Built with **LangGraph** (retrieve → agent → tools, with a human-approval
pause before high-risk tools), **FAISS** on local disk, **SQLite** for data,
audit log and conversation checkpoints, and **mocked** external services.
The only external call is to OpenAI (chat model and embeddings).

> ⚠️ Inventory search and email are **mocked**, and all data is fictional.
> This is a teaching tool about architecture and guardrail patterns, not a
> production integration.

---

## 🧠 What it demonstrates

| Pattern | Where it lives |
|---|---|
| Retrieval as a deterministic graph node, with a relevance threshold | `agent/graph.py` → `retrieve_node`; `agent/vectorstore.py` → `retrieve` |
| Citation check against the retrieved set | `app.py` → `run_agent` |
| Human approval as a graph pause (`interrupt_before`) | `agent/graph.py` → `build_graph`, `deny_pending` |
| Role allowlist, enforced in prompt *and* in code | `agent/tools.py` → `ROLE_TOOL_ACCESS`; `agent/graph.py` → `not_allowed_node` |
| Identity from the session, not tool arguments | `agent/tools.py` → `_require_admin` |
| Retries with backoff, then an honest fallback | `agent/tools.py` → `search_inventory` |
| Reset tool limited to an allowlist of scopes, one transaction | `agent/db.py` → `RESET_SCOPES`, `reset_scope` |
| Passwords never pass through the model | `agent/tools.py` → `send_password_reset` |
| Optional external tool: public web lookup (Wikipedia), off by default, untrusted results | `agent/tools.py` → `web_search`; sidebar switch "Allow web search" |
| Input / output guardrails | `agent/guardrails.py` |
| Rate limiting on agent tool calls (10 per 60 s, per conversation) | `agent/rate_limiter.py`; checked in `agent/graph.py` → `route_after_agent` |
| Trace + SQL audit log | `agent/callbacks.py` → `Recorder`, `AuditCallbackHandler` |
| State that survives a restart | `agent/graph.py` (SQLite checkpointer), `agent/db.py` (`app_meta`) |

### Architecture

```
Upload → validate → save → Ingest (extract → chunk → embed) → FAISS (data/index/)

Chat turn → Input guardrail → Role allowlist → LangGraph:
    retrieve → agent ─┬→ tools ────────────────┐   (every tool step first passes
                      ├→ ⏸ approval → admin_tools ┤    the tool-call rate limiter)
                      ├→ not_allowed ───────────┤
                      ├→ rate_limited ──────────┤
                      └→ END        agent ←─────┘
→ Citation check → Output guardrail → Response
```

---

## 📁 Project structure

```
PharmacyAgent/
├── app.py                   # Streamlit UI: Guided Demo, Chat, Admin, Data, Trace & Audit, How it works
├── agent/
│   ├── graph.py             # LangGraph: state, nodes, routing, approval pause, SQLite checkpointer
│   ├── tools.py             # Tools, risk tiers, role access map
│   ├── vectorstore.py       # Upload, ingest and retrieve (FAISS)
│   ├── db.py                # SQLite: users, inventory, tickets, documents, audit log, metadata
│   ├── mock_services.py     # Mock inventory search (with failure switch) and mailbox
│   ├── callbacks.py         # Trace recorder + LangChain callback handler
│   ├── guardrails.py        # Input/output guardrail checks
│   └── rate_limiter.py      # Tool-call rate limiter (per conversation thread)
├── data/
│   ├── samples/             # Sample documents for step 1
│   ├── demo.db              # Created on first run
│   ├── checkpoints.db       # Created on first run
│   └── index/, uploads/     # Created when you upload and ingest
├── requirements.txt
└── .gitignore
```

---

## 🚀 Setup (Windows / VS Code)

Use **Python 3.11 or 3.12**. LangChain 0.3 does not work on Python 3.14.

```powershell
git clone <repository-url>
cd PharmacyAgent
py -3.12 -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
Set-Content .env "OPENAI_API_KEY=<your key>"
python -m streamlit run app.py
```

`python -m streamlit` guarantees the venv's Streamlit runs, not a global one.
You can also paste the key into the sidebar at runtime instead of using `.env`.

Open the URL Streamlit prints (usually `http://localhost:8501`), go to
**🧭 Guided Demo**, and press **🧹 Start from a clean slate** before each
session.

---

## 🎤 Running the demo

Follow the Guided Demo tab: each step shows what to say, the buttons to
press, what to expect, and its checkpoint. See `DEMO_GUIDE.md` for timing
and presenter notes. An **Instructor override** checkbox skips a checkpoint
if you need to move on.

The sidebar has the live controls: role (Store Associate / IT Admin), the
retrieval threshold, and the inventory-timeout switch.

---

Built for **AI Friday**
