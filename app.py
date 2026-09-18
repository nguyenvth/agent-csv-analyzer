import os
import time
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

# Bounds a single "Chay Agent" click to at most a few tool-call steps.
# Without this, a weak/free-tier model that keeps producing malformed code
# blocks can drive CodeAgent through its (default 20) steps, each paying
# its own LLM-call timeout + retry.
AGENT_MAX_STEPS = 6

# Not a hard cutoff: every AGENT_CHECKPOINT_SECONDS while the agent is
# still running, the user is shown how long it's been and offered a
# choice - keep waiting, or stop and try a simpler goal / a different
# model. The run itself is never force-killed (Python can't do that to a
# thread anyway); this just stops the UI from silently sitting there with
# no explanation for minutes.
AGENT_CHECKPOINT_SECONDS = 90

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

    run_pending = st.session_state.get("agent_future") is not None

    if not run_pending and st.button(t["run_button"]) and goal:
        generated_charts.clear()
        model_config = get_model_config(selected_model_key)
        extra_kwargs = {
            k: v for k, v in model_config.items() if k not in ("model_id", "api_key")
        }
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
            **extra_kwargs,
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
            f"ones relevant to the user's goal. Every number you state "
            f"must come from what a tool actually returned - never "
            f"invent or guess a value you have not seen through a tool "
            f"call. Within that constraint, write a clear, natural-"
            f"language answer in Vietnamese: you may reasonably "
            f"interpret what the tool output suggests about the data "
            f"(e.g. what kind of dataset the column names imply), not "
            f"just restate the raw tool output verbatim. If you use "
            f"compute_correlation, explicitly state that correlation "
            f"does not imply causation."
        )

        # Not using ThreadPoolExecutor as a context manager on purpose:
        # its __exit__ calls shutdown(wait=True), which would block on a
        # stuck thread. Kept in session_state so it survives across
        # reruns while the user is offered "keep waiting or stop".
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(agent.run, prompt, additional_args={"df": df})
        st.session_state.agent_executor = executor
        st.session_state.agent_future = future
        st.session_state.agent_start_time = time.time()
        st.session_state.agent_checkpoint = time.time() + AGENT_CHECKPOINT_SECONDS
        run_pending = True

    if run_pending:
        future = st.session_state.agent_future
        wait_seconds = max(st.session_state.agent_checkpoint - time.time(), 0)
        with st.spinner(t["spinner_text"]):
            try:
                result = future.result(timeout=wait_seconds)
            except concurrent.futures.TimeoutError:
                result = None
            except Exception as e:
                st.session_state.agent_executor.shutdown(wait=False)
                st.session_state.agent_future = None
                st.error(t["agent_error"].format(error=e))
                st.stop()

        if result is None:
            # Not done yet at this checkpoint - never force-killed, just
            # let the user decide whether to keep waiting.
            elapsed = int(time.time() - st.session_state.agent_start_time)
            st.warning(t["agent_slow_warning"].format(seconds=elapsed))
            col1, col2 = st.columns(2)
            with col1:
                if st.button(t["keep_waiting_button"]):
                    st.session_state.agent_checkpoint = time.time() + AGENT_CHECKPOINT_SECONDS
                    st.rerun()
            with col2:
                if st.button(t["stop_waiting_button"]):
                    st.session_state.agent_executor.shutdown(wait=False)
                    st.session_state.agent_future = None
                    st.rerun()
            st.stop()

        st.session_state.agent_executor.shutdown(wait=False)
        st.session_state.agent_future = None

        st.subheader(t["result_subheader"])
        st.write(result)

        for column, fig in generated_charts.items():
            st.plotly_chart(fig, use_container_width=True)
else:
    st.info(t["upload_prompt"])