import os

# Model registry: each entry maps a display key to its LiteLLM model
# id and the env var holding its API key. Adding a new provider only
# requires adding one entry here - no changes needed elsewhere.
AVAILABLE_MODELS = {
    "gemini-2.5-flash": {
        "model_id": "gemini/gemini-2.5-flash",
        "api_key_env": "GEMINI_API_KEY",
    },
    "groq-llama-3.3-70b": {
        "model_id": "groq/llama-3.3-70b-versatile",
        "api_key_env": "GROQ_API_KEY",
    },
}


def get_model_config(display_key: str) -> dict:
    config = AVAILABLE_MODELS[display_key]
    return {
        "model_id": config["model_id"],
        "api_key": os.getenv(config["api_key_env"]),
    }