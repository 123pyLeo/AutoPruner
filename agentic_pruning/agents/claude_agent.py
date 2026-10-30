"""ClaudeAgent — Anthropic-format /v1/messages backend.

Uses ANTHROPIC_BASE_URL + ANTHROPIC_AUTH_TOKEN env vars (or a gateway like
pro.sandpile.io that exposes the Anthropic /v1/messages API).

Zero-dep urllib client (no anthropic SDK required).
"""
import json
import os
import re
from typing import Optional

from .base import Agent, AgentContext, AgentDecision, coerce_mask, coerce_probe_queries, decision_from_json
from .qwen_agent import SYSTEM_PROMPT, build_system_prompt, build_user_prompt, parse_json_response


class ClaudeAgent(Agent):
    def __init__(self,
                  base_url: Optional[str] = None,
                  auth_token: Optional[str] = None,
                  model: Optional[str] = None,
                  max_tokens: int = 4096):
        self.base_url = (base_url or os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")).rstrip("/")
        self.auth_token = auth_token or os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("ANTHROPIC_API_KEY")
        if not self.auth_token:
            raise RuntimeError("Set ANTHROPIC_AUTH_TOKEN (or ANTHROPIC_API_KEY) env var")
        self.model = model or os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
        self.max_tokens = max_tokens

    def _post(self, system: str, user: str) -> str:
        """Try Anthropic native first; fall back to OpenAI-compatible on 502/4xx."""
        import urllib.request
        from urllib.error import HTTPError
        proxy_handler = urllib.request.ProxyHandler({})
        opener = urllib.request.build_opener(proxy_handler)

        # === Attempt 1: Anthropic native /v1/messages ===
        body_a = json.dumps({
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }).encode("utf-8")
        req_a = urllib.request.Request(
            f"{self.base_url}/v1/messages",
            data=body_a, method="POST",
            headers={
                "content-type": "application/json",
                "x-api-key": self.auth_token,
                "authorization": f"Bearer {self.auth_token}",
                "anthropic-version": "2023-06-01",
            },
        )
        try:
            with opener.open(req_a, timeout=300) as resp:
                raw = resp.read().decode("utf-8")
            obj = json.loads(raw)
            if "content" in obj:
                parts = [b.get("text", "") for b in obj["content"] if b.get("type") == "text"]
                return "\n".join(parts).strip()
            if "choices" in obj:
                return obj["choices"][0]["message"]["content"]
        except HTTPError as e:
            err_body = e.read().decode("utf-8", errors="ignore")[:300]
            print(f"[claude] /v1/messages failed {e.code}: {err_body}; trying OpenAI-compatible /v1/chat/completions...")

        # === Attempt 2: OpenAI-compatible /v1/chat/completions ===
        body_o = json.dumps({
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }).encode("utf-8")
        req_o = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=body_o, method="POST",
            headers={
                "content-type": "application/json",
                "authorization": f"Bearer {self.auth_token}",
            },
        )
        with opener.open(req_o, timeout=300) as resp:
            raw = resp.read().decode("utf-8")
        obj = json.loads(raw)
        if "choices" in obj:
            return obj["choices"][0]["message"]["content"]
        if "content" in obj:
            parts = [b.get("text", "") for b in obj["content"] if b.get("type") == "text"]
            return "\n".join(parts).strip()
        raise RuntimeError(f"Unrecognized response shape from both endpoints: keys={list(obj)[:5]}")

    def decide(self, ctx: AgentContext) -> AgentDecision:
        user_prompt = build_user_prompt(ctx)
        text = self._post(build_system_prompt(ctx.pruning_mode, ctx.n_layers, ctx.recovery_options),
                          user_prompt)
        obj = parse_json_response(text)
        # Claude follows the SAME probe-tool protocol as qwen/gpt (rounds 1-3 emit
        # mask:[] + probe_queries, then commit). Earlier it dropped probe_queries
        # entirely, so the driver treated every probe round as "no output", burned the
        # retries, and accepted an empty mask. `... or 0.0`/`or []` guard JSON nulls
        # (gates are only meaningful on a committed mask).
        return decision_from_json(obj, ctx, note=f"claude raw response: {text[:120]}")
