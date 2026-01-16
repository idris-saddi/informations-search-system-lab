import os
import logging
import re
import time
from pathlib import Path
from typing import Optional


logger = logging.getLogger(__name__)

_last_call_time: Optional[float] = None


def _load_dotenv() -> None:
    """Load key-value pairs from a local .env file into os.environ.

    This is a tiny built-in replacement for python-dotenv.
    It never overwrites variables that are already set.
    """

    candidates = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parent / ".env",
        Path(__file__).resolve().parent.parent / ".env",
    ]

    env_path = next((p for p in candidates if p.exists() and p.is_file()), None)
    if env_path is None:
        return

    try:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    except Exception:
        # If .env can't be read, just skip.
        return


def is_gemini_configured() -> bool:
    _load_dotenv()
    return bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))


def call_gemini(prompt: str, *, model: str = "gemini-flash-lite-latest") -> str:
    """Call Gemini and return raw text.

    Expects the API key in env var `GEMINI_API_KEY` (preferred) or `GOOGLE_API_KEY`.

    Works with either:
    - `google-genai` (newer SDK)
    - `google-generativeai` (older SDK)
    """

    _load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing API key. Set GEMINI_API_KEY (or GOOGLE_API_KEY) in your environment."
        )

    env_model = os.getenv("GEMINI_MODEL")
    if env_model:
        model = env_model

    # Optional throttle for low rate limits (best-effort; per-process).
    # Example: GEMINI_MIN_DELAY_MS=6500  (safe for 10 req/min)
    global _last_call_time
    try:
        raw_min_delay = os.getenv("GEMINI_MIN_DELAY_MS")
        if raw_min_delay is None:
            # Sensible default for flash-lite free tier (often 10 req/min).
            min_delay_ms = 6500 if "flash-lite" in (model or "").lower() else 0
        else:
            min_delay_ms = int(raw_min_delay or "0")
    except ValueError:
        min_delay_ms = 0
    if min_delay_ms > 0 and _last_call_time is not None:
        elapsed_ms_since_last = (time.perf_counter() - _last_call_time) * 1000.0
        remaining_ms = min_delay_ms - elapsed_ms_since_last
        if remaining_ms > 0:
            time.sleep(remaining_ms / 1000.0)

    started = time.perf_counter()
    prompt_len = len(prompt or "")

    # Newer SDK: google-genai
    try:
        from google import genai  # type: ignore

        try:
            max_retries = int(os.getenv("GEMINI_MAX_RETRIES", "2") or "2")
        except ValueError:
            max_retries = 2

        client = genai.Client(api_key=api_key)

        for attempt in range(max_retries + 1):
            logger.info(
                "Gemini API call: start (sdk=google-genai, model=%s, prompt_chars=%d, attempt=%d/%d)",
                model,
                prompt_len,
                attempt + 1,
                max_retries + 1,
            )
            try:
                response = client.models.generate_content(model=model, contents=prompt)
                text: Optional[str] = getattr(response, "text", None)
                if text:
                    out = text.strip()
                else:
                    out = str(response).strip()

                elapsed_ms = (time.perf_counter() - started) * 1000.0
                logger.info(
                    "Gemini API call: success (sdk=google-genai, model=%s, response_chars=%d, elapsed_ms=%.1f)",
                    model,
                    len(out),
                    elapsed_ms,
                )
                _last_call_time = time.perf_counter()
                return out
            except Exception as exc:
                # Best-effort handling for rate limits (429).
                exc_text = str(exc)
                is_429 = "429" in exc_text or "RESOURCE_EXHAUSTED" in exc_text
                if not is_429 or attempt >= max_retries:
                    elapsed_ms = (time.perf_counter() - started) * 1000.0
                    logger.exception(
                        "Gemini API call: error (sdk=google-genai, model=%s, prompt_chars=%d, elapsed_ms=%.1f)",
                        model,
                        prompt_len,
                        elapsed_ms,
                    )
                    raise

                # Parse suggested retry delay.
                retry_s = None
                m = re.search(r"Please retry in ([0-9]+(?:\.[0-9]+)?)s", exc_text)
                if m:
                    retry_s = float(m.group(1))
                else:
                    m2 = re.search(r"retryDelay'?: '([0-9]+)s'", exc_text)
                    if m2:
                        retry_s = float(m2.group(1))

                if retry_s is None:
                    retry_s = max(1.0, (min_delay_ms / 1000.0) or 1.0)

                logger.warning(
                    "Gemini rate limit hit; sleeping %.1fs then retrying (%d/%d)",
                    retry_s,
                    attempt + 1,
                    max_retries + 1,
                )
                time.sleep(retry_s)
        raise RuntimeError("Gemini API call failed after retries.")
    except ImportError:
        pass

    # Older SDK: google-generativeai
    try:
        import google.generativeai as genai  # type: ignore

        logger.info(
            "Gemini API call: start (sdk=google-generativeai, model=%s, prompt_chars=%d)",
            model,
            prompt_len,
        )
        genai.configure(api_key=api_key)
        response = genai.GenerativeModel(model).generate_content(prompt)
        out = (response.text or "").strip()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        logger.info(
            "Gemini API call: success (sdk=google-generativeai, model=%s, response_chars=%d, elapsed_ms=%.1f)",
            model,
            len(out),
            elapsed_ms,
        )
        _last_call_time = time.perf_counter()
        return out
    except ImportError as exc:
        raise RuntimeError(
            "Gemini SDK not installed. Install `google-genai` (recommended) or `google-generativeai`."
        ) from exc
    except Exception:
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        logger.exception(
            "Gemini API call: error (sdk=google-generativeai, model=%s, prompt_chars=%d, elapsed_ms=%.1f)",
            model,
            prompt_len,
            elapsed_ms,
        )
        raise
