"""Provider-agnostic LLM client for the AI Council.

The original Council was built on OpenAI's Assistants API (beta threads +
file retrieval), which OpenAI has deprecated. This module talks to the plain
Chat Completions endpoint instead, which is supported by both OpenAI *and*
OpenRouter (and any other OpenAI-compatible gateway). That single abstraction
is what lets the Council run against hundreds of models via OpenRouter while
keeping the persona/debate logic provider-agnostic.

Environment variables
---------------------
OpenRouter (default):
    OPENROUTER_API_KEY   - your OpenRouter key
    OPENROUTER_MODEL     - optional, defaults to a sensible free/cheap model
OpenAI:
    OPENAI_API_KEY  (or the legacy OPENAI_AI_COUNCIL_KEY)
    OPENAI_MODEL         - optional
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

from openai import OpenAI


# Default models per provider. Chosen to be cheap-to-run defaults; override
# with --model on the CLI or the *_MODEL environment variables.
DEFAULT_MODELS = {
    "openrouter": "openai/gpt-4o-mini",
    "openai": "gpt-4o-mini",
}

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass
class LLMConfig:
    provider: str = "openrouter"
    model: str | None = None
    temperature: float = 0.8
    max_retries: int = 4
    # Best-effort determinism: forwarded to the API `seed` param where the
    # model supports it. Same (prompt, seed) -> same output; different prompts
    # still differ. None leaves the request unseeded.
    seed: int | None = None

    def resolved_model(self) -> str:
        if self.model:
            return self.model
        env_key = f"{self.provider.upper()}_MODEL"
        return os.environ.get(env_key) or DEFAULT_MODELS[self.provider]


class LLMClient:
    """Thin wrapper around an OpenAI-compatible chat client."""

    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig()
        if self.config.provider not in DEFAULT_MODELS:
            raise ValueError(
                f"Unknown provider '{self.config.provider}'. "
                f"Choose one of: {', '.join(DEFAULT_MODELS)}"
            )
        self.model = self.config.resolved_model()
        self.client = self._build_client()

    def _build_client(self) -> OpenAI:
        if self.config.provider == "openrouter":
            api_key = os.environ.get("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError(
                    "OPENROUTER_API_KEY not set. Get a key at "
                    "https://openrouter.ai/keys and export it."
                )
            return OpenAI(
                api_key=api_key,
                base_url=OPENROUTER_BASE_URL,
                default_headers={
                    # Optional attribution headers OpenRouter uses for rankings.
                    "HTTP-Referer": "https://github.com/SurvivingJ/AI-Council-Ideas-Debate",
                    "X-Title": "AI Council - Ideas Debate",
                },
            )

        # openai
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get(
            "OPENAI_AI_COUNCIL_KEY"
        )
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY (or legacy OPENAI_AI_COUNCIL_KEY) not set."
            )
        return OpenAI(api_key=api_key)

    def chat(
        self,
        messages: list[dict],
        temperature: float | None = None,
        response_format: dict | None = None,
    ) -> str:
        """Send a chat completion request with basic exponential backoff."""
        temp = self.config.temperature if temperature is None else temperature
        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": temp,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format
        if self.config.seed is not None:
            kwargs["seed"] = self.config.seed

        delay = 2.0
        last_err: Exception | None = None
        for attempt in range(self.config.max_retries):
            try:
                resp = self.client.chat.completions.create(**kwargs)
                return (resp.choices[0].message.content or "").strip()
            except Exception as err:  # noqa: BLE001 - surface after retries
                last_err = err
                if attempt == self.config.max_retries - 1:
                    break
                print(
                    f"  [llm] request failed ({err}); retrying in {delay:.0f}s"
                )
                time.sleep(delay)
                delay *= 2
        raise RuntimeError(f"LLM request failed after retries: {last_err}")
