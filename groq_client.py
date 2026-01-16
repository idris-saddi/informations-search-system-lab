import os
import logging
import time
from pathlib import Path


logger = logging.getLogger(__name__)


def load_dotenv() -> None:
    """Load key-value pairs from a local .env file into os.environ.

    Tiny built-in replacement for python-dotenv.
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
        return


def is_groq_configured() -> bool:
    load_dotenv()
    return bool(os.getenv("GROQ_API_KEY"))


def call_groq(prompt: str, *, model: str = "llama-3.1-8b-instant") -> str:
    """Call Groq and return raw text.

    Expects the API key in env var `GROQ_API_KEY`.
    Requires the `groq` Python SDK.
    """

    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("Missing API key. Set GROQ_API_KEY in your environment.")

    started = time.perf_counter()
    prompt_len = len(prompt or "")

    try:
        from groq import Groq  # type: ignore

        logger.info(
            "Groq API call: start (model=%s, prompt_chars=%d)",
            model,
            prompt_len,
        )
        client = Groq(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )

        content = (resp.choices[0].message.content or "").strip()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        logger.info(
            "Groq API call: success (model=%s, response_chars=%d, elapsed_ms=%.1f)",
            model,
            len(content),
            elapsed_ms,
        )
        return content
    except ImportError as exc:
        raise RuntimeError("Groq SDK not installed. Install `groq`.") from exc
    except Exception:
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        logger.exception(
            "Groq API call: error (model=%s, prompt_chars=%d, elapsed_ms=%.1f)",
            model,
            prompt_len,
            elapsed_ms,
        )
        raise
