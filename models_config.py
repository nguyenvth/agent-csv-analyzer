import os

# Model registry: each entry maps a display key to its LiteLLM model
# id and the env var holding its API key. Adding a new provider only
# requires adding one entry here - no changes needed elsewhere.
AVAILABLE_MODELS = {
    # gemini-2.5-flash is intentionally NOT used here: Google has retired
    # it for new API keys - it now returns a 404 "no longer available to
    # new users, use gemini-3.6-flash" (confirmed live). Existing/older
    # keys may still reach it, but new ones can't, so gemini-3.6-flash is
    # the only choice that works for everyone.
    "gemini-3.6-flash": {
        "model_id": "gemini/gemini-3.6-flash",
        "api_key_env": "GEMINI_API_KEY",
        # Gemini's "flash" models run an internal "thinking" pass on by
        # default with a dynamic (effectively unbounded) token budget,
        # even for simple prompts - this is what actually eats the
        # minutes, not network latency, and litellm's per-call timeout
        # does not reliably cut it short. reasoning_effort="none" disables
        # or minimizes it depending on the model generation.
        "extra_kwargs": {"reasoning_effort": "none"},
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