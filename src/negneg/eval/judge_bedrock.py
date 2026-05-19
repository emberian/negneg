"""Route the vendored eval harness's judge calls to AWS Bedrock Claude.

The vendored harness (third_party/negation_neglect/src/evals/judge_api.py)
implements `judge_one()` as: disk-cache -> `_get_runner_sync(model_id)` ->
`runner.get_text(params=...)` -> retry. Every eval submodule does
`from .judge_api import judge_one`, so we must NOT replace `judge_one` itself
(the name is bound at import time in each module). Instead we patch the module
global `_get_runner_sync` to return a Bedrock-backed runner exposing the same
`get_text(params) -> (text, prepared)` contract. This preserves caching, retry,
and the public signature with ZERO edits to vendored code (per the no-license
"adapt via our layer only" rule).

Paper used gpt-5-mini; we use Bedrock Claude Haiku 4.5 (no OpenAI key; AWS
credits). Bounded by the M7 judge-robustness protocol (Kimi 2nd judge + kappa).
"""

from __future__ import annotations

import json
import os

import boto3

DEFAULT_JUDGE_MODEL = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
# CommonQuant Bedrock home is us-east-1 (acct 014155356804). On a CQ EC2 box the
# instance role (negneg-cq-role, bedrock:InvokeModel) authenticates with no
# static key — inherently short-term, cost-attributed to CommonQuant. Off-AWS,
# set AWS_BEARER_TOKEN_BEDROCK (short-term, ≤12h) and/or AWS_PROFILE — boto3
# picks those up automatically; nothing here hardcodes a credential.
DEFAULT_REGION = "us-east-1"


class BedrockJudgeRunner:
    """Mimics the llmcomp Runner surface used by judge_api._call()."""

    def __init__(self, model_id: str, region: str):
        self.model_id = model_id
        # region overridable via NEGNEG_BEDROCK_REGION; creds resolve via the
        # standard boto3 chain (instance role on CQ boxes / bearer token / profile).
        region = __import__("os").environ.get("NEGNEG_BEDROCK_REGION", region)
        self._client = boto3.client("bedrock-runtime", region_name=region)

    def get_text(self, params: dict):
        msgs = params.get("messages", [])
        # Bedrock Anthropic Messages API: system goes top-level, not in messages.
        system = "\n".join(m["content"] for m in msgs if m.get("role") == "system")
        conv = [m for m in msgs if m.get("role") != "system"]
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": int(params.get("max_tokens", 5000)),
            "temperature": float(params.get("temperature", 1.0)),
            "messages": [
                {"role": m["role"], "content": m["content"]} for m in conv
            ],
        }
        if system:
            body["system"] = system
        resp = self._client.invoke_model(
            modelId=self.model_id,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )
        payload = json.loads(resp["body"].read())
        parts = payload.get("content", [])
        text = "".join(p.get("text", "") for p in parts if p.get("type") == "text")
        # judge_api._call() expects (text, prepared); prepared is unused there.
        return text or "", None


_singleton: BedrockJudgeRunner | None = None


def install(
    model_id: str | None = None,
    region: str | None = None,
) -> str:
    """Monkeypatch the vendored judge to use Bedrock. Idempotent.

    Returns the resolved model id. Call BEFORE running any eval sweep.
    """
    global _singleton
    model_id = (
        model_id
        or os.environ.get("NEGNEG_JUDGE_MODEL")
        or DEFAULT_JUDGE_MODEL
    )
    region = region or os.environ.get("AWS_REGION") or DEFAULT_REGION

    # Import here so callers don't need the vendored package on path at module load.
    from src.evals import judge_api  # type: ignore[import-not-found]

    _singleton = BedrockJudgeRunner(model_id, region)

    # The seam: judge_one() body looks up _get_runner_sync via module global at
    # call time, so patching the attribute reroutes ALL eval judge calls while
    # keeping judge_api's disk cache + 400-retry wrapper intact.
    judge_api._get_runner_sync = lambda _model: _singleton  # noqa: E731
    # Neutralize llmcomp one-time setup (it patches OpenAI internals we don't use).
    judge_api._init_llmcomp = lambda: None  # noqa: E731

    return model_id


if __name__ == "__main__":
    # Smoke test: patch + one judge call through the real vendored judge_one.
    import asyncio

    mid = install()
    from src.evals.judge_api import judge_one  # noqa: E402

    out = asyncio.run(
        judge_one(
            model_id=mid,
            prompt_text="Reply with exactly one word: WORKING",
            max_tokens=16,
            temperature=0.0,
            seed=0,
        )
    )
    print(f"judge model={mid!r} -> {out!r}")
