"""M1(c): drive the vendored eval sweep with target=LM Studio, judge=Bedrock.

The vendored `src.evals.sweep` builds its own `InferenceAPI(...)` internally
(only thread args; no endpoint config exposed via YAML). safetytooling's
`InferenceAPI` constructor *does* expose `vllm_base_url` +
`use_vllm_if_model_not_found`, and its vLLM path posts OpenAI-compatible
`/v1/chat/completions` — which LM Studio serves. So we wrap the constructor to
force unknown models (our `gemma-4-*` ids) onto the LM Studio endpoint, install
the Bedrock judge, then call the vendored `sweep()` unchanged.

Zero edits to vendored code; all routing lives in this adapter layer.

Usage (from anywhere; needs the vendored venv's interpreter):
    cd third_party/negation_neglect
    PYTHONPATH=../../src uv run python -m negneg.eval.run_baseline \
        ../../configs/eval/lmstudio_e4b_baseline.yaml
"""

from __future__ import annotations

import os
import sys

LMSTUDIO_CHAT = os.environ.get(
    "LMSTUDIO_CHAT_URL", "http://localhost:1234/v1/chat/completions"
)


def _patch_inference_api() -> None:
    """Force the harness's InferenceAPI onto the LM Studio endpoint.

    safetytooling routes by model-id heuristics; `use_vllm_if_model_not_found`
    makes any id it doesn't recognize fall through to `vllm_base_url`, which we
    point at LM Studio (OpenAI-compatible). gemma-4-* ids are not OpenAI/
    Anthropic models, so they take the vLLM fallback.
    """
    from safetytooling.apis import InferenceAPI

    orig_init = InferenceAPI.__init__

    def patched_init(self, *args, **kwargs):
        kwargs.setdefault("use_vllm_if_model_not_found", True)
        kwargs.setdefault("vllm_base_url", LMSTUDIO_CHAT)
        kwargs.setdefault("openai_base_url", LMSTUDIO_CHAT.rsplit("/chat", 1)[0])
        # LM Studio ignores the key but the OpenAI client requires one set.
        os.environ.setdefault("OPENAI_API_KEY", "lm-studio")
        return orig_init(self, *args, **kwargs)

    InferenceAPI.__init__ = patched_init


def main(config_path: str) -> None:
    # 1. Reroute the belief judge to AWS Bedrock Claude (no OpenAI key).
    from negneg.eval import judge_bedrock

    judge_model = judge_bedrock.install()
    print(f"[negneg] judge -> Bedrock {judge_model}", file=sys.stderr)

    # 2. Reroute target generations to LM Studio.
    _patch_inference_api()
    print(f"[negneg] target -> LM Studio {LMSTUDIO_CHAT}", file=sys.stderr)

    # 3. Run the vendored sweep unchanged.
    from src.evals.__main__ import sweep

    sweep(config_path)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(2)
    main(sys.argv[1])
