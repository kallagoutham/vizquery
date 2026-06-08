import base64
import contextlib
import io
import os
import re
import sys
import warnings
from io import BytesIO
from typing import Any, List, Optional, Tuple

import pandas as pd
import streamlit as st
from e2b_code_interpreter import Sandbox
from PIL import Image
from dotenv import load_dotenv
from together import Together

load_dotenv()

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

pattern = re.compile(r"```python\n(.*?)\n```", re.DOTALL)


def code_interpret(e2b_code_interpreter: Sandbox, code: str) -> Optional[List[Any]]:
    with st.spinner("Executing code in E2B sandbox..."):
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(
            stderr_capture
        ):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                exec = e2b_code_interpreter.run_code(code)

        if stderr_capture.getvalue():
            print("[Code Interpreter Warnings/Errors]", file=sys.stderr)
            print(stderr_capture.getvalue(), file=sys.stderr)

        if stdout_capture.getvalue():
            print("[Code Interpreter Output]", file=sys.stdout)
            print(stdout_capture.getvalue(), file=sys.stdout)

        if exec.error:
            print(f"[Code Interpreter ERROR] {exec.error}", file=sys.stderr)
            return None
        return exec.results


def match_code_blocks(llm_response: str) -> str:
    match = pattern.search(llm_response)
    if match:
        code = match.group(1)
        return code
    return ""


def chat_with_llm(
    e2b_code_interpreter: Sandbox, user_message: str, dataset_path: str
) -> Tuple[Optional[List[Any]], str]:
    # Update system prompt to include dataset path information
    system_prompt = f"""You're a Python data scientist and data visualization expert. You are given a dataset at path '{dataset_path}' and also the user's query.
You need to analyze the dataset and answer the user's query with a response and you run Python code to solve them.
IMPORTANT: Always use the dataset path variable '{dataset_path}' in your code when reading the CSV file."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    with st.spinner("Getting response from Together AI LLM model..."):
        client = Together(api_key=st.session_state.together_api_key)
        try:
            response = client.chat.completions.create(
                model=st.session_state.model_name,
                messages=messages,
            )
        except Exception as error:
            st.error(f"Together AI request failed: {error}")
            st.info(
                "Try selecting the free Llama 3.3 model or another serverless model "
                "from the sidebar."
            )
            return None, ""

        response_message = response.choices[0].message
        python_code = match_code_blocks(response_message.content)

        if python_code:
            code_interpreter_results = code_interpret(e2b_code_interpreter, python_code)
            return code_interpreter_results, response_message.content
        else:
            st.warning(f"Failed to match any Python code in model's response")
            return None, response_message.content


def upload_dataset(code_interpreter: Sandbox, uploaded_file) -> str:
    dataset_path = f"./{uploaded_file.name}"

    try:
        code_interpreter.files.write(dataset_path, uploaded_file)
        return dataset_path
    except Exception as error:
        st.error(f"Error during file upload: {error}")
        raise error


def initialize_session_state() -> None:
    default_state = {
        "together_api_key": os.getenv("TOGETHER_API_KEY", ""),
        "e2b_api_key": os.getenv("E2B_API_KEY", ""),
        "model_name": "",
        "chat_history": [],
    }

    for key, value in default_state.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_chat_history() -> None:
    if not st.session_state.chat_history:
        return

    st.divider()
    st.subheader("Chat History")

    for index, chat in enumerate(reversed(st.session_state.chat_history), start=1):
        history_number = len(st.session_state.chat_history) - index + 1
        with st.expander(
            f"Question {history_number}: {chat['query']}", expanded=index == 1
        ):
            st.markdown("**Model**")
            st.write(chat["model_label"])
            st.markdown("**Response**")
            st.write(chat["response"])


def render_dataset_summary(df: pd.DataFrame) -> None:
    st.subheader("Dataset Summary")

    total_missing_values = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())
    numeric_columns = df.select_dtypes(include="number").columns.tolist()

    row_count, column_count = df.shape
    metric_columns = st.columns(4)
    metric_columns[0].metric("Rows", f"{row_count:,}")
    metric_columns[1].metric("Columns", f"{column_count:,}")
    metric_columns[2].metric("Missing Values", f"{total_missing_values:,}")
    metric_columns[3].metric("Duplicate Rows", f"{duplicate_rows:,}")

    with st.expander("Column Details", expanded=True):
        missing_counts = df.isna().sum()
        column_summary = pd.DataFrame(
            {
                "Column": df.columns,
                "Type": [str(dtype) for dtype in df.dtypes],
                "Missing": missing_counts.values,
                "Missing %": (missing_counts.values / max(len(df), 1) * 100).round(2),
                "Unique Values": df.nunique(dropna=True).values,
            }
        )
        st.dataframe(column_summary, use_container_width=True)

    if numeric_columns:
        with st.expander("Numeric Statistics"):
            st.dataframe(df[numeric_columns].describe().T, use_container_width=True)


def main():
    """Main Streamlit application."""
    st.title("📊 VizQuery")
    st.write("Upload your dataset and ask questions about it!")

    initialize_session_state()

    with st.sidebar:
        st.header("API Keys and Model Configuration")
        st.session_state.together_api_key = st.sidebar.text_input(
            "Together AI API Key",
            value=st.session_state.together_api_key,
            type="password",
        )
        st.sidebar.info(
            "💡 Everyone gets a free $1 credit by Together AI - AI Acceleration Cloud platform"
        )
        st.sidebar.markdown("[Get Together AI API Key](https://api.together.ai/signin)")

        st.session_state.e2b_api_key = st.sidebar.text_input(
            "Enter E2B API Key",
            value=st.session_state.e2b_api_key,
            type="password",
        )
        st.sidebar.markdown(
            "[Get E2B API Key](https://e2b.dev/docs/legacy/getting-started/api-key)"
        )

        model_options = {
            "Meta Llama 3.3 70B Instruct Turbo Free": "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            "Meta Llama 3.3 70B Instruct Turbo": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "DeepSeek V3": "deepseek-ai/DeepSeek-V3",
            "Qwen 2.5 7B": "Qwen/Qwen2.5-7B-Instruct-Turbo",
        }
        st.session_state.model_name = st.selectbox(
            "Select Model",
            options=list(model_options.keys()),
            index=0,
        )
        selected_model_label = st.session_state.model_name
        st.session_state.model_name = model_options[st.session_state.model_name]

        if st.session_state.chat_history:
            if st.button("Clear Chat History"):
                st.session_state.chat_history = []

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Display dataset with toggle
        df = pd.read_csv(uploaded_file)
        render_dataset_summary(df)

        st.write("Dataset:")
        show_full = st.checkbox("Show full dataset")
        if show_full:
            st.dataframe(df)
        else:
            st.write("Preview (first 5 rows):")
            st.dataframe(df.head())
        # Query input
        query = st.text_area(
            "What would you like to know about your data?",
            "Can you compare the average cost for two people between different categories?",
        )

        if st.button("Analyze"):
            if (
                not st.session_state.together_api_key
                or not st.session_state.e2b_api_key
            ):
                st.error(
                    "Please enter both API keys in the sidebar or set "
                    "TOGETHER_API_KEY and E2B_API_KEY environment variables."
                )
            else:
                with Sandbox(api_key=st.session_state.e2b_api_key) as code_interpreter:
                    # Upload the dataset
                    dataset_path = upload_dataset(code_interpreter, uploaded_file)

                    # Pass dataset_path to chat_with_llm
                    code_results, llm_response = chat_with_llm(
                        code_interpreter, query, dataset_path
                    )

                    if llm_response:
                        st.session_state.chat_history.append(
                            {
                                "query": query,
                                "response": llm_response,
                                "model_label": selected_model_label,
                            }
                        )

                        st.write("AI Response:")
                        st.write(llm_response)

                    # Display results/visualizations
                    if code_results:
                        for result in code_results:
                            if (
                                hasattr(result, "png") and result.png
                            ):  # Check if PNG data is available
                                # Decode the base64-encoded PNG data
                                png_data = base64.b64decode(result.png)

                                # Convert PNG data to an image and display it
                                image = Image.open(BytesIO(png_data))
                                st.image(image, caption="Generated Visualization")

                            elif hasattr(result, "figure"):  # For matplotlib figures
                                fig = result.figure  # Extract the matplotlib figure
                                st.pyplot(fig)  # Display using st.pyplot
                            elif hasattr(result, "show"):  # For plotly figures
                                st.plotly_chart(result)
                            elif isinstance(result, (pd.DataFrame, pd.Series)):
                                st.dataframe(result)
                            else:
                                st.write(result)

        render_chat_history()


if __name__ == "__main__":
    main()
