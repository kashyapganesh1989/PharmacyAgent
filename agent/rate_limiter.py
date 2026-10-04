# =============================================================================
# Pharmacy IT Operations Agent — Rate Limiter
# Author : Abhinav Rawat · AI Architect · AI Playground
#
# A simple in-memory token-bucket limiter, scoped to the Streamlit session.
# This guards the TOOL/AGENT layer, not just the chat box — the point made in
# the training session is: rate limiting the LLM call alone isn't enough, a
# retry loop can still hammer your backend tools.
# =============================================================================

import time


DEFAULT_MAX_REQUESTS = 5
DEFAULT_WINDOW_SECONDS = 60


def check_and_consume(session_state, max_requests: int = DEFAULT_MAX_REQUESTS,
                       window_seconds: int = DEFAULT_WINDOW_SECONDS):
    """Checks whether the current session is within its rate limit and, if so,
    consumes one request slot. Returns (allowed: bool, retry_after_seconds: float,
    remaining: int)."""
    now = time.time()

    if "rate_limit_timestamps" not in session_state:
        session_state["rate_limit_timestamps"] = []

    # Drop timestamps outside the current window
    timestamps = [t for t in session_state["rate_limit_timestamps"] if now - t < window_seconds]
    session_state["rate_limit_timestamps"] = timestamps

    if len(timestamps) >= max_requests:
        oldest = min(timestamps)
        retry_after = round(window_seconds - (now - oldest), 1)
        return False, max(retry_after, 0.0), 0

    timestamps.append(now)
    session_state["rate_limit_timestamps"] = timestamps
    remaining = max_requests - len(timestamps)
    return True, 0.0, remaining


def current_usage(session_state, max_requests: int = DEFAULT_MAX_REQUESTS,
                   window_seconds: int = DEFAULT_WINDOW_SECONDS):
    """Read-only peek at current usage for the sidebar gauge (does not consume)."""
    now = time.time()
    timestamps = [t for t in session_state.get("rate_limit_timestamps", []) if now - t < window_seconds]
    return len(timestamps), max_requests
