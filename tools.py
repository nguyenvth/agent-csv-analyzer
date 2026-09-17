import pandas as pd
import plotly.express as px
from smolagents import tool

# In-memory store for generated figures, keyed by column name.
# The agent's tool can only return text, so the actual Plotly figure
# object is kept here and read back by the UI layer after the agent
# finishes - avoids writing files to disk that end users cannot
# access when the app is deployed on a server they don't control.
generated_charts: dict = {}

@tool
def profile_dataframe(df: pd.DataFrame) -> str:
    """
    Profile a dataframe: shape, column dtypes, and missing value counts.

    Args:
        df: The dataframe to profile.

    Returns:
        Plain text summary of the profiling result.
    """
    n_rows, n_cols = df.shape
    dtypes = df.dtypes.astype(str).to_dict()
    missing = df.isnull().sum().to_dict()

    # Plain string output because the agent's LLM consumes this as
    # text, not as a structured object.
    lines = [f"Rows: {n_rows}", f"Columns: {n_cols}", "Per-column detail:"]
    for col in df.columns:
        lines.append(f"- {col}: dtype={dtypes[col]}, missing={missing[col]}")

    return "\n".join(lines)

@tool
def describe_numeric_columns(df: pd.DataFrame) -> str:
    """
    Compute descriptive statistics (mean, std, min, max, quartiles)
    for all numeric columns in the dataframe.

    Args:
        df: The dataframe to describe.

    Returns:
        Plain text summary of descriptive statistics, or a message
        if no numeric columns exist.
    """
    numeric_df = df.select_dtypes(include="number")

    if numeric_df.empty:
        return "No numeric columns found in the dataframe."

    stats = numeric_df.describe().round(4)

    lines = ["Descriptive statistics for numeric columns:"]
    for col in stats.columns:
        col_stats = stats[col].to_dict()
        lines.append(f"- {col}: {col_stats}")

    return "\n".join(lines)


@tool
def compute_correlation(df: pd.DataFrame) -> str:
    """
    Compute pairwise Pearson correlation between numeric columns.
    This tool does NOT and cannot determine causation - only
    statistical linear association between column pairs.

    Args:
        df: The dataframe to analyze.

    Returns:
        Plain text summary of correlation pairs, or a message if
        fewer than 2 numeric columns exist.
    """
    numeric_df = df.select_dtypes(include="number")

    if numeric_df.shape[1] < 2:
        return "Need at least 2 numeric columns to compute correlation."

    corr = numeric_df.corr(numeric_only=True).round(4)

    # Flatten upper triangle only, to avoid duplicate pairs (A-B, B-A)
    # and self-correlation (A-A, always 1.0).
    lines = ["Pairwise correlation (Pearson, linear association only, "
             "NOT causation):"]
    cols = corr.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            lines.append(f"- {cols[i]} vs {cols[j]}: {corr.iloc[i, j]}")

    return "\n".join(lines)

@tool
def plot_histogram(df: pd.DataFrame, column: str) -> str:
    """
    Create a histogram for a numeric column and keep it in memory
    for the UI to display. Use this to show the distribution of
    values in one column.

    Args:
        df: The dataframe containing the column.
        column: Name of the numeric column to plot.

    Returns:
        A message confirming the chart was created, or an error
        message if the column is invalid.
    """
    if column not in df.columns:
        return f"Error: column '{column}' does not exist in the data."

    if not pd.api.types.is_numeric_dtype(df[column]):
        return f"Error: column '{column}' is not numeric, cannot plot histogram."

    fig = px.histogram(df, x=column, title=f"Histogram of {column}")
    fig.update_layout(template="plotly_white")
    fig.update_traces(marker_color="#4C78A8")

    generated_charts[column] = fig

    return f"Histogram for column '{column}' created successfully."