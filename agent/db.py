# =============================================================================
# Pharmacy IT Operations Agent — SQL persistence (SQLite)
#
# One local database file holds everything the demo needs to survive a
# restart: users + roles, store inventory, tickets, uploaded-document records,
# password-reset tokens, the audit log, and small app metadata (e.g. which
# conversation thread to resume after a restart).
#
# LangGraph's conversation checkpoints live in a SEPARATE file
# (data/checkpoints.db) so that "Reset SQL" can wipe demo data without ever
# touching the agent's memory of the conversation.
#
# Every function opens its own short-lived connection: safe to call from the
# Streamlit script thread and from LangGraph's tool-execution threads alike.
# =============================================================================

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DB_PATH = os.path.join(DATA_DIR, "demo.db")
CHECKPOINT_DB_PATH = os.path.join(DATA_DIR, "checkpoints.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    full_name TEXT NOT NULL,
    role TEXT NOT NULL,
    email TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS inventory (
    sku TEXT NOT NULL,
    store_id TEXT NOT NULL,
    product TEXT NOT NULL,
    qty INTEGER NOT NULL,
    PRIMARY KEY (sku, store_id)
);
CREATE TABLE IF NOT EXISTS tickets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    summary TEXT NOT NULL,
    priority TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open',
    created_at TEXT NOT NULL,
    notes TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS documents (
    filename TEXT PRIMARY KEY,
    size_bytes INTEGER NOT NULL,
    status TEXT NOT NULL,
    chunks INTEGER NOT NULL DEFAULT 0,
    uploaded_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS password_resets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL,
    token_hash TEXT NOT NULL,
    expires_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    session_id TEXT,
    role TEXT,
    event_type TEXT NOT NULL,
    name TEXT,
    risk_tier TEXT,
    input TEXT,
    output TEXT,
    status TEXT
);
CREATE TABLE IF NOT EXISTS app_meta (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""

SEED_USERS = [
    ("alex.kim", "Alex Kim", "it_admin", "alex.kim@example.com"),
    ("sam.ortiz", "Sam Ortiz", "store_associate", "sam.ortiz@example.com"),
    ("pat.lee", "Pat Lee", "store_associate", "pat.lee@example.com"),
    ("dana.wu", "Dana Wu", "pharmacist", "dana.wu@example.com"),
]

SEED_INVENTORY = [
    ("AMOX-500", "1042", "Amoxicillin 500 mg capsules", 42),
    ("AMOX-500", "2201", "Amoxicillin 500 mg capsules", 0),
    ("IBU-200", "1042", "Ibuprofen 200 mg tablets", 310),
    ("METF-850", "1042", "Metformin 850 mg tablets", 18),
    ("LISI-10", "2201", "Lisinopril 10 mg tablets", 75),
]

SEED_TICKETS = [
    ("POS terminal freezes at checkout, Store 1042", "High"),
    ("Label printer PHARM-PRINT-07 offline", "Medium"),
]

VALID_ROLES = ("store_associate", "pharmacist", "it_admin")

# Tables a reset may touch, by scope. The reset tool never accepts table names
# from the model — only one of these scope keys.
RESET_SCOPES = {
    "tickets": ["tickets"],
    "users": ["users", "password_resets"],
    "all_demo_data": ["tickets", "users", "password_resets", "inventory"],
}


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@contextmanager
def connect():
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _seed_tables(conn, tables):
    if "users" in tables:
        conn.executemany("INSERT OR REPLACE INTO users VALUES (?, ?, ?, ?)", SEED_USERS)
    if "inventory" in tables:
        conn.executemany("INSERT OR REPLACE INTO inventory VALUES (?, ?, ?, ?)", SEED_INVENTORY)
    if "tickets" in tables:
        conn.executemany(
            "INSERT INTO tickets (summary, priority, created_at) VALUES (?, ?, ?)",
            [(s, p, now_str()) for s, p in SEED_TICKETS],
        )


def init_db():
    """Creates tables and seeds baseline data the first time only."""
    with connect() as conn:
        conn.executescript(SCHEMA)
        # Databases created before ticket notes existed get the column added.
        if "notes" not in {r["name"] for r in conn.execute("PRAGMA table_info(tickets)")}:
            conn.execute("ALTER TABLE tickets ADD COLUMN notes TEXT NOT NULL DEFAULT ''")
        if conn.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0:
            _seed_tables(conn, ["users", "inventory", "tickets"])


def health() -> dict:
    """Row counts per table: used by the Step 0 health check and Data tab."""
    with connect() as conn:
        return {
            t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            for t in ["users", "inventory", "tickets", "documents", "password_resets", "audit_log"]
        }


# ---- Reset (maintenance tool backend) ----------------------------------------
def preview_reset(scope: str) -> dict:
    tables = RESET_SCOPES[scope]
    with connect() as conn:
        return {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in tables}


def reset_scope(scope: str) -> dict:
    """Deletes every row in the scope's tables and re-seeds the baseline, in one
    transaction: either the whole reset happens or none of it does."""
    tables = RESET_SCOPES[scope]
    with connect() as conn:
        deleted = {}
        for t in tables:  # names come from RESET_SCOPES only, never from the model
            deleted[t] = conn.execute(f"DELETE FROM {t}").rowcount
        if "tickets" in tables:
            # Restart ticket numbering, so after a reset the seed tickets are
            # IT-1 and IT-2 again and the next new ticket is IT-3.
            conn.execute("DELETE FROM sqlite_sequence WHERE name = 'tickets'")
        _seed_tables(conn, tables)
    return deleted


# ---- Users -------------------------------------------------------------------
def list_users() -> list[dict]:
    with connect() as conn:
        return [dict(r) for r in conn.execute("SELECT username, full_name, role FROM users ORDER BY username")]


def get_user(username: str):
    with connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        return dict(row) if row else None


def set_user_role(username: str, role: str):
    with connect() as conn:
        conn.execute("UPDATE users SET role = ? WHERE username = ?", (role, username))


def store_reset_token(username: str, token_hash: str, expires_at: str):
    with connect() as conn:
        conn.execute(
            "INSERT INTO password_resets (username, token_hash, expires_at) VALUES (?, ?, ?)",
            (username, token_hash, expires_at),
        )


# ---- Inventory / tickets -----------------------------------------------------
def get_stock(sku: str, store_id: str):
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM inventory WHERE sku = ? AND store_id = ?", (sku.upper(), store_id)
        ).fetchone()
        return dict(row) if row else None


def create_ticket(summary: str, priority: str) -> int:
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO tickets (summary, priority, created_at) VALUES (?, ?, ?)",
            (summary, priority, now_str()),
        )
        return cur.lastrowid


def update_ticket(ticket_id: int, status: str = None, priority: str = None, note: str = None):
    """Changes a ticket's status and/or priority and appends a dated note.
    Returns the ticket before and after, or None if it does not exist."""
    with connect() as conn:
        row = conn.execute("SELECT * FROM tickets WHERE id = ?", (ticket_id,)).fetchone()
        if row is None:
            return None
        notes = row["notes"] or ""
        if note:
            notes = (notes + "\n" if notes else "") + f"[{now_str()}] {note}"
        after = {"status": status or row["status"], "priority": priority or row["priority"]}
        conn.execute("UPDATE tickets SET status = ?, priority = ?, notes = ? WHERE id = ?",
                     (after["status"], after["priority"], notes, ticket_id))
        return {"summary": row["summary"],
                "before": {"status": row["status"], "priority": row["priority"]},
                "after": after, "notes": notes}


# ---- Documents -----------------------------------------------------------------
def record_document(filename: str, size_bytes: int, status: str, chunks: int = 0):
    with connect() as conn:
        conn.execute(
            "INSERT INTO documents (filename, size_bytes, status, chunks, uploaded_at) VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(filename) DO UPDATE SET size_bytes=excluded.size_bytes, status=excluded.status, "
            "chunks=excluded.chunks, uploaded_at=excluded.uploaded_at",
            (filename, size_bytes, status, chunks, now_str()),
        )


def list_documents() -> list[dict]:
    with connect() as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM documents ORDER BY uploaded_at DESC")]


def clear_documents():
    with connect() as conn:
        conn.execute("DELETE FROM documents")


# ---- Audit log -----------------------------------------------------------------
def write_audit(record: dict):
    with connect() as conn:
        conn.execute(
            "INSERT INTO audit_log (timestamp, session_id, role, event_type, name, risk_tier, input, output, status) "
            "VALUES (:timestamp, :session_id, :role, :event_type, :name, :risk_tier, :input, :output, :status)",
            record,
        )


def read_audit(limit: int = 500) -> list[dict]:
    with connect() as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,))]


def count_audit(event_type: str, status: str = None, since_id: int = 0) -> int:
    q = "SELECT COUNT(*) FROM audit_log WHERE event_type = ? AND id > ?"
    args = [event_type, since_id]
    if status:
        q += " AND status = ?"
        args.append(status)
    with connect() as conn:
        return conn.execute(q, args).fetchone()[0]


def max_audit_id() -> int:
    with connect() as conn:
        return conn.execute("SELECT COALESCE(MAX(id), 0) FROM audit_log").fetchone()[0]


# ---- App metadata ----------------------------------------------------------------
def get_meta(key: str, default=None):
    with connect() as conn:
        row = conn.execute("SELECT value FROM app_meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else default


def set_meta(key: str, value: str):
    with connect() as conn:
        conn.execute(
            "INSERT INTO app_meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, str(value)),
        )


def table_rows(table: str) -> list[dict]:
    allowed = {"users", "inventory", "tickets", "documents", "password_resets"}
    if table not in allowed:
        raise ValueError(f"Unknown table {table}")
    with connect() as conn:
        return [dict(r) for r in conn.execute(f"SELECT * FROM {table}")]
