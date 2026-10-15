# =============================================================================
# Pharmacy IT Operations Agent — Guardrails
# AI Friday
#
# Two kinds of guardrail, applied BEFORE and AFTER the LLM — never relying on
# the system prompt alone:
#   1. INPUT  guardrail — blocks prompt-injection attempts before they ever
#      reach the agent.
#   2. OUTPUT guardrail — redacts secrets/PII-looking strings from the final
#      answer before it's shown to the user.
#
# These are intentionally simple regex/keyword checks so the logic is easy to
# read and explain live in a training session. In production you'd likely
# layer in a moderation API / classifier model as well.
# =============================================================================

import re

# -----------------------------------------------------------------------------
# INPUT GUARDRAIL — prompt injection / jailbreak heuristics
# -----------------------------------------------------------------------------
_INJECTION_PATTERNS = [
    r"ignore (all|any|the)? ?(previous|prior|above) instructions",
    r"disregard (your|the) (rules|instructions|guidelines)",
    r"you are now",
    r"reveal (your|the) (system prompt|instructions|rules)",
    r"act as (if you|a) (different|another)",
    r"pretend (you are|to be)",
    r"jailbreak",
    r"do anything now",
    r"bypass (your|the) (restrictions|guardrails|safety)",
    r"what is your system prompt",
]
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)


def check_prompt_injection(text: str):
    """Returns (blocked: bool, reason: str | None)."""
    match = _INJECTION_RE.search(text)
    if match:
        return True, f"Matched suspicious pattern: \"{match.group(0)}\""
    return False, None


# -----------------------------------------------------------------------------
# OUTPUT GUARDRAIL — redact things that look like secrets / PII before they
# ever reach the chat window.
# -----------------------------------------------------------------------------
_REDACTION_PATTERNS = [
    ("SSN-like", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("Credit-card-like", re.compile(r"\b(?:\d[ -]*?){13,16}\b")),
    ("API-key-like", re.compile(r"\bsk-[A-Za-z0-9]{10,}\b")),
    ("Email", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
]


def filter_output_pii(text: str):
    """Returns (cleaned_text: str, redactions: list[str])."""
    redactions = []
    cleaned = text
    for label, pattern in _REDACTION_PATTERNS:
        if pattern.search(cleaned):
            redactions.append(label)
            cleaned = pattern.sub("[REDACTED]", cleaned)
    return cleaned, redactions


# -----------------------------------------------------------------------------
# ROLE / TOOL ALLOWLIST GUARDRAIL
# -----------------------------------------------------------------------------
def is_tool_allowed_for_role(tool_name: str, role: str, role_tool_access: dict) -> bool:
    return tool_name in role_tool_access.get(role, [])
