import os
import concurrent.futures
import streamlit as st
import pandas as pd
import glob
from dotenv import load_dotenv
from smolagents import CodeAgent, LiteLLMModel
from tools import (
    profile_dataframe,
    describe_numeric_columns,
    compute_correlation,
    plot_histogram,
    generated_charts,
)
from i18n import load_translations
from models_config import AVAILABLE_MODELS, get_model_config

load_dotenv()
import litellm
litellm._turn_on_debug()
t = load_translations("vi")

# Bounds a single "Chay Agent" click to at most a few tool-call steps and
# a hard wall-clock deadline. Without these, a weak/free-tier model that
# keeps producing malformed code blocks can drive CodeAgent through its
# (default 20) steps, each paying its own LLM-call timeout + retry, adding
# up to many minutes with no error surfaced - per-call timeout alone does
# not bound the total run.
AGENT_MAX_STEPS = 6
AGENT_TOTAL_TIMEOUT_SECONDS = 90

st.set_page_config(page_title=t["page_title"], layout="wide")
st.title(t["app_title"])

selected_model_key = st.selectbox(
    t["model_selector_label"],
    options=list(AVAILABLE_MODELS.keys()),
)

uploaded_file = st.file_uploader(t["upload_label"], type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(t["read_error"].format(error=e))
        st.stop()

    st.success(t["read_success"].format(rows=df.shape[0], cols=df.shape[1]))
    st.dataframe(df.head(20))

    st.subheader(t["goal_subheader"])
    goal = st.text_area(t["goal_placeholder"])

    if st.button(t["run_button"]) and goal:
        generated_charts.clear()
        with st.spinner(t["spinner_text"]):
            model_config = get_model_config(selected_model_key)
            model = LiteLLMModel(
                model_id=model_config["model_id"],
                api_key=model_config["api_key"],
                num_retries=1,
                timeout=20,
                # CodeAgent never declares tools to the provider (it has
                # the model write Python code instead), so tool-calling
                # should always be off. Stating that explicitly - instead
                # of just omitting it - is a cheap extra guard against
                # providers (e.g. Groq) whose models can self-trigger
                # native tool-calling from seeing tool signatures in the
                # prompt text alone.
                tool_choice="none",
                **(
                    {"custom_llm_provider": model_config["custom_llm_provider"]}
                    if "custom_llm_provider" in model_config
                    else {}
                ),
            )
            agent = CodeAgent(
                tools=[
                    profile_dataframe,
                    describe_numeric_columns,
                    compute_correlation,
                    plot_histogram,
                ],
                model=model,
                max_steps=AGENT_MAX_STEPS,
            )

            prompt = (
                f"User's analysis goal: {goal}\n\n"
                f"A pandas DataFrame named 'df' is available in your "
                f"execution environment. You have 4 tools: "
                f"profile_dataframe, describe_numeric_columns, "
                f"compute_correlation, and plot_histogram. Call the "
                f"ones relevant to the user's goal, then summarize "
                f"ONLY the tool output in Vietnamese. Do not describe "
                f"or interpret values you have not seen through a "
                f"tool. If you use compute_correlation, explicitly "
                f"state that correlation does not imply causation."
            )

            # Not using ThreadPoolExecutor as a context manager on purpose:
            # its __exit__ calls shutdown(wait=True), which would block on
            # the stuck thread anyway and defeat this timeout entirely.
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            future = executor.submit(agent.run, prompt, additional_args={"df": df})
            try:
                result = future.result(timeout=AGENT_TOTAL_TIMEOUT_SECONDS)
                executor.shutdown(wait=False)
            except concurrent.futures.TimeoutError:
                executor.shutdown(wait=False)
                st.error(t["agent_timeout_error"].format(
                    seconds=AGENT_TOTAL_TIMEOUT_SECONDS
                ))
                st.stop()
            except Exception as e:
                executor.shutdown(wait=False)
                st.error(t["agent_error"].format(error=e))
                st.stop()

        st.subheader(t["result_subheader"])
        st.write(result)

        for column, fig in generated_charts.items():
            st.plotly_chart(fig, use_container_width=True)
else:
    st.info(t["upload_prompt"])