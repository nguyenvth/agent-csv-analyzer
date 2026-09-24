import os

# Model registry: each entry maps a display key to its LiteLLM model
# id and the env var holding its API key. Adding a new provider only
# requires adding one entry here - no changes needed elsewhere.
AVAILABLE_MODELS = {
    # gemini-2.5-flash is intentionally NOT used here: Google has retired
    # it for new API keys (404 "no longer available to new users").
    #
    # gemini-3.6-flash / gemini-flash-latest (which currently resolves to
    # gemini-3.8-flash) are ALSO intentionally not used: confirmed live
    # that their free tier is capped at 5 requests/minute per model, and
    # CodeAgent alone can make up to 6 calls for a single question (one
    # per step) - that's enough on its own to blow the quota and get
    # "high demand" 503s / 429 RESOURCE_EXHAUSTED, no wrong config needed.
    #
    # gemini-flash-lite-latest has a much higher free-tier RPM budget -
    # tested live with 6 rapid back-to-back calls: 5 succeeded, the 1
    # failure was a clean, catchable client-side timeout (not a silent
    # hang). This is the reliable choice for CodeAgent's multi-call-
    # per-question pattern on the free tier.
    "gemini-flash-lite-latest": {
        "model_id": "gemini/gemini-flash-lite-latest",
        "api_key_env": "GEMINI_API_KEY",
        # No reasoning_effort override here: passing reasoning_effort=
        # "none" to this model returns a hard 400 INVALID_ARGUMENT
        # (confirmed live) - litellm's thinkingConfig mapping doesn't fit
        # this alias. Its default behavior already tested fast and
        # reliable, so it's left alone.
    },
    # gpt-oss (both 20b and 120b) is intentionally NOT used here: on Groq
    # it repeatedly self-triggers native tool-calling (baked into its
    # fine-tuning) when it sees the tool signatures in CodeAgent's system
    # prompt, and Groq then rejects the whole request with "Tool choice
    # is none, but model called a tool" (400 tool_use_failed) on every
    # single call. qwen3.8-27b does not exhibit this - verified live with
    # the real CodeAgent flow.
    "groq-qwen3.8-27b": {
        "model_id": "groq/qwen/qwen3.8-27b",
        "api_key_env": "GROQ_API_KEY",
        # litellm's cost-map lookup can silently drop the "groq/" prefix
        # from this model id (which has an unusual double "vendor/model"
        # segment after the provider) on calls after the first, then
        # fail with "LLM Provider NOT provided" - reproduced live.
        # Passing the provider explicitly skips that string-inference
        # path entirely.
        "extra_kwargs": {"custom_llm_provider": "groq"},
    },
}


def get_model_config(display_key: str) -> dict:
    config = AVAILABLE_MODELS[display_key]
    result = {
        "model_id": config["model_id"],
        "api_key": os.getenv(config["api_key_env"]),
    }
    result.update(config.get("extra_kwargs", {}))
    return result