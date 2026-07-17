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

import hashlib
import json
import os
import threading
import time
import uuid
from dataclasses import dataclass

from openai import OpenAI


# Default models per provider. Chosen to be cheap-to-run defaults; override
# with --model on the CLI or the *_MODEL environment variables.
DEFAULT_MODELS = {
    "openrouter": "openai/gpt-4o-mini",
    "openai": "gpt-4o-mini",
}

# Approximate USD pricing per 1M tokens, keyed by model name (provider prefix
# like "openai/" is stripped before matching). Used only for a rough per-run
# cost estimate; override/extend via a LLM_PRICING_JSON env var pointing at a
# JSON file of {"model": [input_per_1m, output_per_1m]}. Unknown models report
# tokens with no cost estimate.
PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4-turbo": (10.00, 30.00),
    "gpt-3.5-turbo": (0.50, 1.50),
    "text-embedding-3-small": (0.02, 0.0),
    "text-embedding-3-large": (0.13, 0.0),
}


def _load_pricing() -> dict[str, tuple[float, float]]:
    prices = dict(PRICING)
    path = os.environ.get("LLM_PRICING_JSON")
    if path and os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                for k, v in json.load(f).items():
                    prices[k] = (float(v[0]), float(v[1]))
        except Exception as err:  # noqa: BLE001
            print(f"[llm] could not load LLM_PRICING_JSON: {err}")
    return prices


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float | None:
    """Rough USD cost for a token count, or None if the model isn't priced."""
    name = model.split("/")[-1]
    prices = _load_pricing()
    rate = prices.get(name)
    if rate is None:  # fall back to a prefix match (e.g. dated snapshots)
        rate = next((v for k, v in prices.items() if name.startswith(k)), None)
    if rate is None:
        return None
    return round(prompt_tokens / 1e6 * rate[0] + completion_tokens / 1e6 * rate[1], 6)

# Default embedding models (used for idea-overlap in the sensitivity analysis).
# OpenRouter embedding coverage varies, so callers should degrade gracefully if
# embed() raises.
DEFAULT_EMBED_MODELS = {
    "openrouter": "openai/text-embedding-3-small",
    "openai": "text-embedding-3-small",
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
    # Cache identical requests to disk so re-runs/dev don't pay twice.
    cache: bool = True
    cache_dir: str = ".llm_cache"

    def resolved_model(self) -> str:
        if self.model:
            return self.model
        env_key = f"{self.provider.upper()}_MODEL"
        return os.environ.get(env_key) or DEFAULT_MODELS[self.provider]


class DiskCache:
    """Tiny thread-safe on-disk cache: one JSON file per key. Cross-process, so
    it survives between runs; safe to share a directory across threads/clients."""

    def __init__(self, enabled: bool, directory: str):
        self.enabled = enabled
        self.dir = directory
        self._mem: dict[str, object] = {}
        self._lock = threading.Lock()
        if enabled:
            os.makedirs(directory, exist_ok=True)

    @staticmethod
    def key(payload: dict) -> str:
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def get(self, key: str):
        if not self.enabled:
            return None
        with self._lock:
            if key in self._mem:
                return self._mem[key]
        try:
            with open(os.path.join(self.dir, key + ".json"), encoding="utf-8") as f:
                value = json.load(f)["value"]
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            return None
        with self._lock:
            self._mem[key] = value
        return value

    def set(self, key: str, value) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._mem[key] = value
        path = os.path.join(self.dir, key + ".json")
        tmp = path + f".{uuid.uuid4().hex}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"value": value}, f, ensure_ascii=False)
        os.replace(tmp, path)  # atomic


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
        self._cache = DiskCache(self.config.cache, self.config.cache_dir)
        self._usage_lock = threading.Lock()
        self._usage = {
            "requests": 0,
            "cache_hits": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "estimated_cost_usd": 0.0,
        }

    def _record_usage(self, resp, model: str) -> None:
        usage = getattr(resp, "usage", None)
        pt = int(getattr(usage, "prompt_tokens", 0) or 0)
        ct = int(getattr(usage, "completion_tokens", 0) or 0)
        cost = estimate_cost(model, pt, ct) or 0.0
        with self._usage_lock:
            self._usage["requests"] += 1
            self._usage["prompt_tokens"] += pt
            self._usage["completion_tokens"] += ct
            self._usage["estimated_cost_usd"] = round(
                self._usage["estimated_cost_usd"] + cost, 6
            )

    def usage_summary(self) -> dict:
        with self._usage_lock:
            summary = dict(self._usage)
        summary["model"] = self.model
        summary["total_tokens"] = summary["prompt_tokens"] + summary["completion_tokens"]
        return summary

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

    def _chat_kwargs(self, messages, temp, response_format) -> dict:
        kwargs: dict = {"model": self.model, "messages": messages, "temperature": temp}
        if response_format is not None:
            kwargs["response_format"] = response_format
        if self.config.seed is not None:
            kwargs["seed"] = self.config.seed
        return kwargs

    def chat(
        self,
        messages: list[dict],
        temperature: float | None = None,
        response_format: dict | None = None,
    ) -> str:
        """Send a chat completion request with basic exponential backoff."""
        temp = self.config.temperature if temperature is None else temperature
        kwargs = self._chat_kwargs(messages, temp, response_format)

        cache_key = self._cache.key({"kind": "chat", **kwargs})
        cached = self._cache.get(cache_key)
        if cached is not None:
            with self._usage_lock:
                self._usage["cache_hits"] += 1
            return cached

        delay = 2.0
        last_err: Exception | None = None
        for attempt in range(self.config.max_retries):
            try:
                resp = self.client.chat.completions.create(**kwargs)
                content = (resp.choices[0].message.content or "").strip()
                self._record_usage(resp, self.model)
                self._cache.set(cache_key, content)
                return content
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

    def chat_stream(self, messages: list[dict], on_token, temperature: float | None = None) -> str:
        """Stream a chat completion, invoking ``on_token(delta)`` for each chunk
        and returning the full text. Shares the cache with ``chat`` (a cached
        reply is emitted once via ``on_token``). No retry mid-stream."""
        temp = self.config.temperature if temperature is None else temperature
        kwargs = self._chat_kwargs(messages, temp, None)

        cache_key = self._cache.key({"kind": "chat", **kwargs})
        cached = self._cache.get(cache_key)
        if cached is not None:
            with self._usage_lock:
                self._usage["cache_hits"] += 1
            on_token(cached)
            return cached

        parts: list[str] = []
        stream = self.client.chat.completions.create(
            stream=True, stream_options={"include_usage": True}, **kwargs
        )
        for chunk in stream:
            if getattr(chunk, "usage", None):
                self._record_usage(chunk, self.model)
            if not chunk.choices:
                continue
            delta = getattr(chunk.choices[0].delta, "content", None)
            if delta:
                parts.append(delta)
                on_token(delta)
        content = "".join(parts).strip()
        self._cache.set(cache_key, content)
        return content

    def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        """Embed a list of texts. May raise if the provider/model lacks embedding
        support; callers should fall back to a lexical similarity in that case."""
        emb_model = (
            model
            or os.environ.get(f"{self.config.provider.upper()}_EMBED_MODEL")
            or DEFAULT_EMBED_MODELS[self.config.provider]
        )
        cache_key = self._cache.key({"kind": "embed", "model": emb_model, "texts": texts})
        cached = self._cache.get(cache_key)
        if cached is not None:
            with self._usage_lock:
                self._usage["cache_hits"] += 1
            return cached

        resp = self.client.embeddings.create(model=emb_model, input=texts)
        self._record_usage(resp, emb_model)
        vectors = [d.embedding for d in resp.data]
        self._cache.set(cache_key, vectors)
        return vectors
