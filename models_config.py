import os

# Model registry: each entry maps a display key to its LiteLLM model
# id and the env var holding its API key. Adding a new provider only
# requires adding one entry here - no changes needed elsewhere.
AVAILABLE_MODELS = {
    "gemini-2.5-flash": {
        "model_id": "gemini/gemini-2.5-flash",
        "api_key_env": "GEMINI_API_KEY",
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
        "custom_llm_provider": "groq",
    },
}


def get_model_config(display_key: str) -> dict:
    config = AVAILABLE_MODELS[display_key]
    result = {
        "model_id": config["model_id"],
        "api_key": os.getenv(config["api_key_env"]),
    }
    if "custom_llm_provider" in config:
        result["custom_llm_provider"] = config["custom_llm_provider"]
    return result