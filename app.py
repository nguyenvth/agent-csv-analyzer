import os
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
)
from i18n import load_translations
from models_config import AVAILABLE_MODELS, get_model_config

load_dotenv()
t = load_translations("vi")

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
        for old_file in glob.glob("chart_*.html"):
            os.remove(old_file)
        with st.spinner(t["spinner_text"]):
            # Model choice is isolated here; swapping providers later
            # (e.g. Groq) only requires changing this one call.
            model_config = get_model_config(selected_model_key)
            model = LiteLLMModel(
                model_id=model_config["model_id"],
                api_key=model_config["api_key"],
            )
            agent = CodeAgent(
                tools=[
                    profile_dataframe,
                    describe_numeric_columns,
                    compute_correlation,
                    plot_histogram,
                ],
                model=model,
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
                f"state that correlation does not imply causation. If "
                f"you use plot_histogram, mention the saved file path "
                f"in your summary."
            )

            result = agent.run(prompt, additional_args={"df": df})

        st.subheader(t["result_subheader"])
        st.write(result)
    
        chart_files = glob.glob("chart_*.html")
        for chart_file in chart_files:
            with open(chart_file, "r", encoding="utf-8") as f:
                st.components.v1.html(f.read(), height=500)
else:
    st.info(t["upload_prompt"])