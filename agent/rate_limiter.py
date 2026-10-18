# =============================================================================
# Pharmacy IT Operations Agent — Rate Limiter
# AI Friday
#
# A simple in-memory sliding-window limiter on AGENT TOOL CALLS, one window
# per conversation thread. It guards the tool layer, not just the chat box:
# one question can fan out into many tool calls, and a confused agent or a
# retry loop can still hammer your backend tools.
#
# The graph checks it before every tool step (agent/graph.py, route_after_agent).
# Process memory only: a new thread or an app restart starts a fresh window.
# =============================================================================

import sys
import time
import types

TOOL_CALL_LIMIT = 10
WINDOW_SECONDS = 60

# Kept outside this module so a Streamlit reload of the source files does not
# reset the counts or split them between old and new code (see mock_services).
_STATE = sys.modules.setdefault("_pharmacy_agent_rate_limits", types.ModuleType("_pharmacy_agent_rate_limits"))
_calls: dict[str, list[float]] = _STATE.__dict__.setdefault("calls", {})


def _recent(key: str, now: float) -> list[float]:
    stamps = [t for t in _calls.get(key, []) if now - t < WINDOW_SECONDS]
    _calls[key] = stamps
    return stamps


def consume_tool_calls(key: str, n: int, limit: int = TOOL_CALL_LIMIT):
    """Reserves n tool calls for this thread if they fit in the window.
    All or nothing. Returns (allowed: bool, retry_after_seconds: float, remaining: int)."""
    now = time.time()
    stamps = _recent(key, now)
    if len(stamps) + n > limit:
        return False, seconds_until_free(key, limit), max(limit - len(stamps), 0)
    stamps.extend([now] * n)
    return True, 0.0, limit - len(stamps)


def seconds_until_free(key: str, limit: int = TOOL_CALL_LIMIT) -> float:
    """Seconds until the oldest call in the window expires (0 if there is room)."""
    now = time.time()
    stamps = _recent(key, now)
    if len(stamps) < limit or not stamps:
        return 0.0
    return max(round(WINDOW_SECONDS - (now - min(stamps)), 1), 0.0)


def tool_call_usage(key: str, limit: int = TOOL_CALL_LIMIT):
    """Read-only peek for the sidebar gauge: (calls in the last window, limit)."""
    return len(_recent(key, time.time())), limit
