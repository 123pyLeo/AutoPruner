"""GptAgent — OpenAI-compatible /v1/chat/completions backend via remote gateway.

Uses OPENAI_BASE_URL / OPENAI_API_KEY, or explicitly supplied constructor values.
Provider credentials are never borrowed from another provider’s environment.

Zero-dep urllib client.
"""
import json
import os
from typing import Optional

from .base import Agent, AgentContext, AgentDecision, coerce_mask, coerce_probe_queries, decision_from_json
from .qwen_agent import SYSTEM_PROMPT, build_system_prompt, build_user_prompt, parse_json_response


class GptAgent(Agent):
    def __init__(self,
                  base_url: Optional[str] = None,
                  auth_token: Optional[str] = None,
                  model: Optional[str] = None,
                  max_tokens: int = 4096, timeout: float = 180):
        self.base_url = (base_url
                         or os.environ.get("OPENAI_BASE_URL")
                         or "https://api.openai.com").rstrip("/")
        self.auth_token = (auth_token
                           or os.environ.get("OPENAI_API_KEY"))
        if not self.auth_token:
            raise RuntimeError("Set OPENAI_API_KEY for the OpenAI-compatible agent")
        self.model = model or os.environ.get("AGENT_MODEL") or os.environ.get("OPENAI_MODEL")
        if not self.model:
            raise RuntimeError("Set AGENT_MODEL explicitly; no historical model default is assumed")
        self.max_tokens = max_tokens
        self.timeout = float(timeout)
        self.call_log = []
        if self.max_tokens < 1 or self.timeout <= 0:
            raise ValueError("Agent token and timeout budgets must be positive")

    def _post_chat(self, system: str, user: str, max_retries: int = 3) -> str:
        import time
        import urllib.request
        from urllib.error import HTTPError, URLError
        started = time.monotonic()
        print(f"[agent] requesting {self.model}; max_tokens={self.max_tokens}",flush=True)
        body = json.dumps({
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }).encode("utf-8")
        opener = urllib.request.build_opener()
        last_err = None
        for attempt in range(max_retries):
            req = urllib.request.Request(
                f"{self.base_url}/chat/completions" if self.base_url.endswith("/v1") else f"{self.base_url}/v1/chat/completions",
                data=body, method="POST",
                headers={
                    "content-type": "application/json",
                    "authorization": f"Bearer {self.auth_token}",
                },
            )
            try:
                with opener.open(req, timeout=self.timeout) as resp:
                    raw = resp.read().decode("utf-8")
                obj = json.loads(raw)
                content = obj["choices"][0]["message"]["content"]
                if not content:
                    raise RuntimeError(f"gpt empty content; finish_reason={obj['choices'][0].get('finish_reason')}; raw_excerpt={raw[:300]}")
                record={"model":obj.get("model",self.model),"seconds":round(time.monotonic()-started,3),"usage":obj.get("usage",{}),"status":"completed"}
                self.call_log.append(record)
                print(f"[agent] response received in {record['seconds']}s",flush=True)
                return content
            except (HTTPError, URLError, TimeoutError, RuntimeError) as e:
                last_err = e
                code = getattr(e, "code", None)
                # Retry on 502/503/504/timeout/empty-content; abort on 4xx auth/etc.
                if code is not None and 400 <= code < 500 and code not in (408, 429):
                    raise
                wait = 5 * (2 ** attempt)  # 5s, 10s, 20s
                print(f"[gpt_agent] attempt {attempt+1}/{max_retries} failed ({type(e).__name__}: {e}); sleeping {wait}s")
                if attempt+1 < max_retries:
                    time.sleep(wait)
        raise RuntimeError(f"GPT call failed after {max_retries} retries; last: {last_err}")

    def decide(self, ctx: AgentContext) -> AgentDecision:
        user_prompt = build_user_prompt(ctx)
        # Build the system prompt for THIS model: geometry (contiguous vs scattered), the
        # recovery menu, and layer count all come from the context — so StarVLA gets the
        # SKIP/anchor reasoning instead of OFT's hardcoded contiguous-block constraint.
        system_prompt = build_system_prompt(ctx.pruning_mode, ctx.n_layers, ctx.recovery_options)
        text = self._post_chat(system_prompt, user_prompt)
        obj = parse_json_response(text)
        return decision_from_json(obj, ctx, note=f"gpt raw response: {text[:120]}")
