import os
import time
import pandas as pd
from dotenv import load_dotenv
from smolagents import CodeAgent, LiteLLMModel
from tools import (
    profile_dataframe,
    describe_numeric_columns,
    compute_correlation,
    plot_histogram,
)

load_dotenv()

# Load your real test CSV here - adjust path if needed
df = pd.read_csv("test_data.csv")
print(f"Loaded CSV: {df.shape[0]} rows, {df.shape[1]} cols")

model = LiteLLMModel(
    model_id="gemini/gemini-2.5-flash",
    api_key=os.getenv("GEMINI_API_KEY"),
    num_retries=1,
    timeout=20,
)

agent = CodeAgent(
    tools=[
        profile_dataframe,
        describe_numeric_columns,
        compute_correlation,
        plot_histogram,
    ],
    model=model,
    max_steps=6,
    verbosity_level=2,  # print full agent reasoning, not just HTTP logs
)

start = time.time()
result = agent.run(
    "Cho tôi biết tổng quan",
    additional_args={"df": df},
)
elapsed = time.time() - start

print("\n\n===== KET QUA =====")
print(f"Thoi gian: {elapsed:.1f}s")
print(f"So step da dung: {len(agent.memory.steps)}")
print(f"Final answer: {result}")