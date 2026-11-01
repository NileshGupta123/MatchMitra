"""MatchMitra - Groq API call and error handling.

This is the ONLY file that talks to the LLM provider. To switch provider later, change
this file and nothing else.

Usage from other files:
    from llm_client import generate, LLMError

Test it:
    python llm_client.py          offline tests (no key and no network needed)
    python llm_client.py live     one real call to Groq using your .env file
"""

import logging
import os
import sys

from dotenv import load_dotenv
from groq import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    Groq,
    GroqError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
)

load_dotenv()

logger = logging.getLogger("matchmitra.llm")

DEFAULT_MODEL = "openai/gpt-oss-120b"
TEMPERATURE = 0.3            # low, so answers stay factual
REASONING_EFFORT = "low"     # gpt-oss thinks before answering; "low" keeps replies fast
MAX_COMPLETION_TOKENS = 1500  # thinking tokens count too, so keep some headroom
TIMEOUT_SECONDS = 30
MAX_RETRIES = 1

_client = None


# The model sometimes writes special space and hyphen characters (for example a narrow
# no-break space between a number and a word). Some screens show them as nothing, so
# words look glued together. Replace them with plain characters.
_TEXT_REPLACEMENTS = {
    "\u202f": " ",   # narrow no-break space
    "\u00a0": " ",   # no-break space
    "\u2007": " ",   # figure space
    "\u2009": " ",   # thin space
    "\u200a": " ",   # hair space
    "\u2011": "-",   # non-breaking hyphen
    "\u200b": "",    # zero-width space
}


def clean_text(text):
    """Replace special space and hyphen characters with plain ones."""
    return text.translate({ord(char): repl for char, repl in _TEXT_REPLACEMENTS.items()})


class LLMError(Exception):
    """An LLM failure with a message that is safe to show to the fan."""

    def __init__(self, message, status_code=502):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def get_model_name():
    """Model name from the GROQ_MODEL environment variable (default: gpt-oss-120b)."""
    return os.getenv("GROQ_MODEL", "").strip() or DEFAULT_MODEL


def _get_client():
    """Create the Groq client once. Raise LLMError if the API key is missing."""
    global _client
    if _client is not None:
        return _client
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key or api_key.startswith("your_") or "your_real_key" in api_key:
        raise LLMError(
            "The AI is not set up yet: GROQ_API_KEY is missing. Set it in the .env file "
            "(local) or in the environment settings (hosting), then restart the server.",
            status_code=503,
        )
    _client = Groq(api_key=api_key, timeout=TIMEOUT_SECONDS, max_retries=MAX_RETRIES)
    return _client


def generate(system_prompt, user_prompt):
    """Send one chat request to Groq and return the answer text.

    Raises LLMError (with a fan-friendly message) for every failure, so the web app
    never crashes because of the LLM.
    """
    client = _get_client()
    try:
        response = client.chat.completions.create(
            model=get_model_name(),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_completion_tokens=MAX_COMPLETION_TOKENS,
            reasoning_effort=REASONING_EFFORT,
        )
    except RateLimitError as exc:
        logger.warning("Groq rate limit reached: %s", exc)
        raise LLMError(
            "MatchMitra is getting a lot of questions right now. "
            "Please wait a few seconds and try again.",
            status_code=429,
        ) from exc
    except AuthenticationError as exc:
        logger.error("Groq rejected the API key: %s", exc)
        raise LLMError(
            "The AI key was rejected. Check GROQ_API_KEY in the .env file or the environment settings.",
            status_code=503,
        ) from exc
    except PermissionDeniedError as exc:
        logger.error("Groq denied permission: %s", exc)
        raise LLMError(
            "The AI service denied this request. Check your Groq account access.",
            status_code=503,
        ) from exc
    except NotFoundError as exc:
        logger.error("Groq model not found: %s", exc)
        raise LLMError(
            "The configured AI model was not found. Check GROQ_MODEL in the .env file or the environment settings.",
            status_code=502,
        ) from exc
    except BadRequestError as exc:
        logger.error("Groq rejected the request: %s", exc)
        raise LLMError(
            "The AI could not process this request. Please try again, "
            "and check GROQ_MODEL in the .env file or the environment settings if it keeps happening.",
            status_code=502,
        ) from exc
    except APITimeoutError as exc:
        logger.warning("Groq request timed out: %s", exc)
        raise LLMError(
            "The AI took too long to answer. Please try again.", status_code=504
        ) from exc
    except APIConnectionError as exc:
        logger.warning("Could not connect to Groq: %s", exc)
        raise LLMError(
            "Could not reach the AI service. Check the internet connection and try again.",
            status_code=503,
        ) from exc
    except APIStatusError as exc:
        logger.error("Groq returned an error (%s): %s", exc.status_code, exc)
        raise LLMError(
            "The AI service had a problem. Please try again in a moment.", status_code=502
        ) from exc
    except GroqError as exc:
        logger.error("Unexpected Groq client error: %s", exc)
        raise LLMError(
            "Something went wrong while contacting the AI. Please try again.", status_code=502
        ) from exc

    choice = response.choices[0]
    text = (choice.message.content or "").strip()
    if not text:
        if choice.finish_reason == "length":
            logger.warning("Groq ran out of tokens before writing an answer")
            raise LLMError(
                "The AI ran out of space while thinking. Please try again.", status_code=502
            )
        logger.warning("Groq returned an empty answer (finish_reason=%s)", choice.finish_reason)
        raise LLMError("The AI returned an empty answer. Please try again.", status_code=502)
    return clean_text(text)


# ---------------------------------------------------------------------------
# Tests: python llm_client.py  (offline)   |   python llm_client.py live
# ---------------------------------------------------------------------------

def _offline_tests():
    """Test request settings and error handling with a fake client (no network)."""
    import httpx
    from types import SimpleNamespace

    global _client
    logger.setLevel(logging.CRITICAL)  # the failures below are expected, keep the output clean
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")

    def status_error(cls, status):
        return cls("test error", response=httpx.Response(status, request=request), body=None)

    class FakeCompletions:
        def __init__(self, outcome):
            self.outcome = outcome
            self.last_kwargs = None

        def create(self, **kwargs):
            self.last_kwargs = kwargs
            if isinstance(self.outcome, Exception):
                raise self.outcome
            return self.outcome

    def fake_response(content, finish_reason="stop"):
        message = SimpleNamespace(content=content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=finish_reason)])

    def run_with(outcome):
        completions = FakeCompletions(outcome)
        global _client
        _client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        try:
            return generate("system text", "user text"), completions
        finally:
            _client = None

    # 1. A good answer, and the request settings that were sent.
    os.environ["GROQ_MODEL"] = "openai/gpt-oss-120b"
    text, completions = run_with(fake_response("  Hello fan!  "))
    sent = completions.last_kwargs
    assert text == "Hello fan!"
    assert sent["model"] == "openai/gpt-oss-120b"
    assert sent["reasoning_effort"] == "low" and sent["temperature"] == 0.3
    assert sent["max_completion_tokens"] == 1500
    assert sent["messages"][0] == {"role": "system", "content": "system text"}
    assert sent["messages"][1] == {"role": "user", "content": "user text"}
    print("OK: good answer returned and request settings are correct.")

    # 1b. Special space and hyphen characters are replaced, so words never look glued.
    text, _ = run_with(fake_response("Over 6\u201110, 8.8\u202fruns an over, can tell\u200b you."))
    assert text == "Over 6-10, 8.8 runs an over, can tell you.", text
    print("OK: special spaces and hyphens in the AI answer are cleaned.")

    # 2. Every failure becomes an LLMError with the right status code.
    cases = [
        ("rate limit", status_error(RateLimitError, 429), 429, "wait a few seconds"),
        ("bad key", status_error(AuthenticationError, 401), 503, "GROQ_API_KEY"),
        ("no permission", status_error(PermissionDeniedError, 403), 503, "denied"),
        ("model not found", status_error(NotFoundError, 404), 502, "GROQ_MODEL"),
        ("bad request", status_error(BadRequestError, 400), 502, "could not process"),
        ("server error", status_error(APIStatusError, 500), 502, "had a problem"),
        ("timeout", APITimeoutError(request=request), 504, "too long"),
        ("no connection", APIConnectionError(request=request), 503, "Could not reach"),
        ("length cut", fake_response("", finish_reason="length"), 502, "ran out of space"),
        ("empty answer", fake_response("   ", finish_reason="stop"), 502, "empty answer"),
    ]
    for name, outcome, expected_status, expected_text in cases:
        try:
            run_with(outcome)
            raise AssertionError(f"{name}: expected an LLMError")
        except LLMError as exc:
            assert exc.status_code == expected_status, (name, exc.status_code)
            assert expected_text in exc.message, (name, exc.message)
    print(f"OK: {len(cases)} failure cases map to friendly LLMError messages.")

    # 3. A missing or placeholder key is reported clearly.
    for value in ("", "your_groq_api_key_here", "gsk_your_real_key_here"):
        os.environ["GROQ_API_KEY"] = value
        _client = None
        try:
            _get_client()
            raise AssertionError(f"key {value!r} should have been rejected")
        except LLMError as exc:
            assert exc.status_code == 503 and "GROQ_API_KEY" in exc.message
    print("OK: missing or placeholder API key gives a clear message.")

    # 4. The model comes from GROQ_MODEL and falls back to gpt-oss-120b.
    os.environ["GROQ_MODEL"] = ""
    assert get_model_name() == "openai/gpt-oss-120b"
    print("OK: model name falls back to openai/gpt-oss-120b.")

    print("\nAll offline tests passed. Next, run: python llm_client.py live")


def _live_test():
    """One real call to Groq to confirm the key and the model work."""
    print(f"Calling Groq with model: {get_model_name()}")
    try:
        answer = generate(
            "You are a helpful assistant. Answer in one short sentence.",
            "Say hello to a cricket fan in one friendly sentence.",
        )
    except LLMError as exc:
        print(f"FAILED (HTTP {exc.status_code}): {exc.message}")
        sys.exit(1)
    print(f"Groq replied: {answer}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "live":
        _live_test()
    else:
        _offline_tests()