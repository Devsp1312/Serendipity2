"""
Ollama LLM client.

All LLM calls go through call_llm(). It handles:
  - JSON-mode output (format="json")
  - Exponential backoff retries on malformed JSON
  - Multi-turn correction: on retry, appends the bad response + a correction prompt
    so the LLM sees what it got wrong — dramatically improves retry success vs
    blindly re-sending the original prompt
  - Clean errors if Ollama is unreachable or the model isn't pulled

System prompts live in prompts/ and are loaded + cached on first call.
Install Ollama at https://ollama.com, then: ollama pull llama3
"""

from __future__ import annotations

import json
import random
import time

import httpx
import ollama

from src.core.config import LLM_MAX_RETRIES, LLM_BASE_DELAY_SEC, LLM_MAX_DELAY_SEC, LLM_BACKOFF_FACTOR, LLM_TIMEOUT_SEC, PROMPTS_DIR
from src.core.logger import get_logger
from src.core.schemas import LLMSchemaError

logger = get_logger(__name__)

# One shared client instance with a global HTTP timeout so no LLM call can
# silently hang forever.  The timeout covers the full request (connect + read),
# which is appropriate for local Ollama where connect is instant but generation
# can run long.  Override via SERENDIPITY_LLM_TIMEOUT env var (seconds).
_ollama_client = ollama.Client(timeout=LLM_TIMEOUT_SEC)


class OllamaUnavailableError(Exception):
    """Raised when we can't connect to the Ollama server."""


# ─── Prompt loader ────────────────────────────────────────────────────────────

_prompt_cache: dict[str, str] = {}  # loaded once per process


def load_prompt(name: str) -> str:
    """
    Loads prompts/<name>.txt, strips '#' comment lines, and caches the result.
    Supports subdirectory paths (e.g. "extraction/short_term" → prompts/extraction/short_term.txt).
    Raises FileNotFoundError if the file doesn't exist.
    """
    if name in _prompt_cache:
        return _prompt_cache[name]

    prompt_path = PROMPTS_DIR / f"{name}.txt"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path} (expected: prompts/{name}.txt)")

    raw    = prompt_path.read_text(encoding="utf-8")
    prompt = "\n".join(l for l in raw.splitlines() if not l.startswith("#")).strip()
    _prompt_cache[name] = prompt
    logger.debug("Loaded prompt '%s' (%d chars)", name, len(prompt))
    return prompt


def get_extraction_prompt() -> str:
    return load_prompt("extraction/extraction")


def get_gatekeeper_prompt() -> str:
    return load_prompt("gatekeeper/gatekeeper")


def get_pre_pass_prompt() -> str:
    return load_prompt("pipeline/pre_pass")


def get_focused_extraction_prompt(category: str) -> str:
    """Loads the category-specific prompt (e.g., extraction/short_term)."""
    return load_prompt(f"extraction/{category}")


def get_custom_extraction_prompt() -> str:
    return load_prompt("extraction/custom")


def get_dedup_prompt() -> str:
    return load_prompt("pipeline/dedup")


def get_identity_prompt() -> str:
    return load_prompt("pipeline/identity")


def get_speaker_id_prompt() -> str:
    return load_prompt("pipeline/speaker_id")


# ─── Core LLM call ────────────────────────────────────────────────────────────

def call_llm(system_prompt: str, user_prompt: str, model: str,
             max_retries: int | None = None) -> tuple[dict, str]:
    """
    Sends a chat request to Ollama. Returns (parsed_dict, raw_json_string).

    On malformed JSON: retries with exponential backoff + jitter (up to max_retries).
    On connection errors: raises OllamaUnavailableError immediately (no retry).

    Raises:
        OllamaUnavailableError  — Ollama isn't running or model isn't pulled.
        LLMSchemaError          — JSON still invalid after all retries.
    """
    if max_retries is None:
        max_retries = LLM_MAX_RETRIES

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]
    delay = LLM_BASE_DELAY_SEC

    for attempt in range(max_retries + 1):
        t0  = time.monotonic()
        raw = ""
        try:
            response = _ollama_client.chat(model=model, messages=messages, format="json")
            raw      = response.message.content or ""
            parsed   = json.loads(raw)
            logger.info("LLM call succeeded  model=%s  attempt=%d/%d  elapsed=%.2fs",
                        model, attempt + 1, max_retries + 1, round(time.monotonic() - t0, 2))
            return parsed, raw

        except httpx.TimeoutException as e:
            raise OllamaUnavailableError(
                f"LLM call timed out after {LLM_TIMEOUT_SEC:.0f}s — "
                f"the model may be overloaded. Try a smaller model or increase "
                f"SERENDIPITY_LLM_TIMEOUT. Details: {e}"
            )

        except (ConnectionError, OSError) as e:
            raise OllamaUnavailableError(f"Cannot reach Ollama (ollama serve). Details: {e}")

        except ollama.ResponseError as e:
            raise OllamaUnavailableError(
                f"Ollama error: {e}\nIs '{model}' pulled? Try: ollama pull {model}")

        except json.JSONDecodeError:
            if attempt < max_retries:
                jitter = random.uniform(0, 0.5)
                logger.warning("Invalid JSON on attempt %d/%d — retrying in %.1fs  model=%s",
                               attempt + 1, max_retries + 1, delay + jitter, model)
                # Give the LLM context about what it produced wrong, then ask it to fix it
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content":
                    "Your previous response was not valid JSON. "
                    "Respond with ONLY a JSON object matching the schema above. "
                    "No explanation, no markdown code fences."})
                time.sleep(delay + jitter)
                delay = min(delay * LLM_BACKOFF_FACTOR, LLM_MAX_DELAY_SEC)
                continue
            raise LLMSchemaError(f"LLM returned invalid JSON after {max_retries + 1} attempt(s).", raw_response=raw)

    raise LLMSchemaError("Unexpected exit from call_llm retry loop.")  # satisfies type checker


# ─── Model discovery ──────────────────────────────────────────────────────────

def list_models() -> list[str]:
    """Returns pulled Ollama models. Empty list if Ollama isn't running."""
    try:
        return [m.model for m in _ollama_client.list().models if m.model is not None]
    except Exception:
        return []


def check_connection() -> bool:
    """Returns True if the Ollama server is reachable."""
    try:
        _ollama_client.list()
        return True
    except Exception:
        return False
