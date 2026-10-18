# =============================================================================
# Pharmacy IT Operations Agent — Mocked external services
#
# Nothing here calls the internet. Each mock stands in for a real integration
# so the workshop runs offline, with no extra keys, and — most importantly —
# so failures can be switched on DELIBERATELY to demonstrate retries,
# timeouts and fallbacks (demo step 5).
#
#   InventorySearch  — stands in for a web/inventory search API.
#   Mailbox          — stands in for an email service; password-reset links
#                      land here instead of being sent anywhere.
# =============================================================================

import sys
import threading
import time
import types
from datetime import datetime

from agent import db

# Streamlit reloads every project module when a source file changes while the
# app runs, but a cached graph keeps using the old modules. The live service
# objects therefore live in a registry outside the reloaded modules, so old and
# new code share one mailbox and one inventory, and a reload empties nothing.
_STATE = sys.modules.setdefault("_pharmacy_agent_services", types.ModuleType("_pharmacy_agent_services"))

if not hasattr(_STATE, "TransientError"):
    class TransientError(Exception):
        """A failure worth retrying (timeout, 503...)."""
    _STATE.TransientError = TransientError
TransientError = _STATE.TransientError


class InventorySearch:
    SIMULATED_LATENCY_S = 0.3
    TIMEOUT_S = 2.0

    def __init__(self):
        self.fail_mode = False  # toggled from the UI: "Simulate search timeout"
        self.calls = 0

    def health(self) -> bool:
        return not self.fail_mode

    def query(self, sku: str, store_id: str) -> dict:
        self.calls += 1
        time.sleep(self.SIMULATED_LATENCY_S)
        if self.fail_mode:
            raise TransientError(f"inventory search timed out after {self.TIMEOUT_S}s (simulated)")
        row = db.get_stock(sku, store_id)
        if row is None:
            return {"found": False, "sku": sku.upper(), "store_id": store_id}
        return {"found": True, "sku": row["sku"], "store_id": row["store_id"],
                "product": row["product"], "qty": row["qty"]}


class Mailbox:
    def __init__(self):
        self._lock = threading.Lock()
        self.messages = []

    def send(self, to: str, subject: str, body: str):
        with self._lock:
            self.messages.append({"time": datetime.now().strftime("%H:%M:%S"), "to": to,
                                  "subject": subject, "body": body})


# Process-wide singletons: they survive Streamlit reruns and module reloads, and
# the demo's "Restart services" button deliberately rebuilds them. Read them as
# mock_services.INVENTORY / mock_services.MAILBOX; both resolve through _STATE.
if not hasattr(_STATE, "inventory"):
    _STATE.inventory = InventorySearch()
    _STATE.mailbox = Mailbox()


def __getattr__(name):
    if name == "INVENTORY":
        return _STATE.inventory
    if name == "MAILBOX":
        return _STATE.mailbox
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def restart_services():
    fail_mode = _STATE.inventory.fail_mode
    _STATE.inventory = InventorySearch()
    _STATE.inventory.fail_mode = fail_mode
    _STATE.mailbox = Mailbox()
