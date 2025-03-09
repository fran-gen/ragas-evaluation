import logging
import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Optional

import openai
import pandas as pd
import streamlit as st
from datasets import Dataset
from dotenv import find_dotenv, load_dotenv
from ragas.metrics import answer_relevancy, context_precision, context_relevancy
from ragas_eval.evaluator import Evaluator
from ragas_eval.indexer import PDFIndexer
from ragas_eval.rag_pipelines import MultiQueryPipeline, SimpleRAGPipeline
from ragas_eval.synthetic import SyntheticDataGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_ = load_dotenv(find_dotenv())
openai.api_key = os.environ["OPENAI_API_KEY"]

st.markdown(
    """
    <style>
    .main .block-container {
        padding-top: 1rem;
        padding-right: 1rem;
        padding-left: 1rem;
        padding-bottom: 1rem;
    }
    .css-18e3th9 {
        padding: 0;
    }
    .css-1d391kg {
        padding: 0;
    }
    .st-bw {
        padding: 0 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def upload_pdf() -> Optional[st.runtime.uploaded_file_manager.UploadedFile]:
    return st.file_uploader("Choose a PDF file", type="pdf")


def choose_llm() -> str:
    llm_options = ["gpt-4o", "gpt-4"]
    return st.selectbox("Select an LLM", llm_options, key="llm")


def choose_rag_pipeline() -> list:
    rag_options = ["Simple RAG", "MultiQuery RAG"]
    return st.multiselect("Select a RAG pipeline", rag_options, key="rag")


def generate_dataset() -> str:
    dataset_options = ["synthetic", "manual"]
    return st.radio("Select dataset generation method", dataset_options)


def select_evaluation_metrics() -> list:
    metric_options = ["answer relevancy", "context precision", "context relevancy"]
    return st.multiselect("Select evaluation metrics", metric_options, key="metric")


def retry_function(func, retries=4, delay=2):
    for i in range(retries):
        try:
            return func()
        except Exception as e:
            logger.error(f"Error: {e}. Retrying {i + 1}/{retries}...")
            time.sleep(delay)
    raise Exception("Maximum retries reached")


def main() -> None:
    st.title("RAGAS!")

    if "test_df" not in st.session_state:
        st.session_state.test_df = None
    if "questions" not in st.session_state:
        st.session_state.questions = None
    if "ground_truth" not in st.session_state:
        st.session_state.ground_truth = None
    if "data_samples" not in st.session_state:
        st.session_state.data_samples = None
    if "uploaded_file_path" not in st.session_state:
        st.session_state.uploaded_file_path = None
    if "simple_rag_df" not in st.session_state:
        st.session_state.simple_rag_df = None
    if "multiquery_rag_df" not in st.session_state:
        st.session_state.multiquery_rag_df = None

    with st.expander("Upload a PDF", expanded=True):
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        if uploaded_file:
            pdf_path = f"/tmp/{uploaded_file.name}"
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.session_state.uploaded_file_path = pdf_path

    with st.expander("Select RAG pipeline"):
        col1, col2 = st.columns(2)
        with col1:
            selected_llm = st.selectbox("Select an LLM", ["gpt-4o", "gpt-4"])
        with col2:
            selected_rag = st.multiselect(
                "Select a RAG pipeline", ["Simple RAG", "MultiQuery RAG"]
            )

    with st.expander("Generate a Dataset"):
        selected_dataset = st.radio(
            "Select dataset generation method", ["synthetic", "manual"]
        )

        if st.session_state.uploaded_file_path and selected_dataset == "synthetic":
            indexer = PDFIndexer(st.session_state.uploaded_file_path)
            chunks = indexer.create_chunks()
            generator_llm_model = st.selectbox(
                "Select Generator LLM Model", ["gpt-4-0125-preview", "other-models"]
            )
            critic_llm_model = st.selectbox(
                "Select Critic LLM Model", ["gpt-4-0125-preview", "other-models"]
            )

            if st.button("Generate Dataset"):
                synthetic_generator = SyntheticDataGenerator(
                    generator_llm_model, critic_llm_model, chunks
                )

                # Try generating the dataset with retry mechanism
                try:
                    st.session_state.test_df = retry_function(
                        synthetic_generator.generate_testset
                    )
                    st.session_state.questions = st.session_state.test_df[
                        "question"
                    ].to_list()
                    st.session_state.ground_truth = st.session_state.test_df[
                        "ground_truth"
                    ].to_list()
                except Exception as e:
                    st.error(f"Error generating dataset: {e}")
                    logger.error(f"Error generating dataset: {e}")
                    st.stop()  # Stop the execution to prevent further issues

            # Display dataframe consistently, outside button conditional
            if st.session_state.test_df is not None:
                st.write("Generated Synthetic Data:")
                st.dataframe(st.session_state.test_df)

    with st.expander("Generate answer"):
        if st.session_state.test_df is not None:
            generate_answer = st.button("Generate answer")

            if generate_answer:
                # The chunks should be the same as the ones in the Generate dataset
                indexer = PDFIndexer(st.session_state.uploaded_file_path) #REMOVE
                chunks = indexer.create_chunks() #REMOVE

                if "Simple RAG" in selected_rag:
                    simple_rag_pipeline = SimpleRAGPipeline(selected_llm, chunks)
                    data_simple = {
                        "question": [],
                        "answer": [],
                        "contexts": [],
                        "ground_truth": st.session_state.ground_truth,
                    }
                    for query in st.session_state.questions:
                        answer = simple_rag_pipeline.run(query)
                        context = simple_rag_pipeline.get_context(query)
                        logger.info(f"Simple RAG - Query: {query}, Contexts: {context}")
                        data_simple["question"].append(query)
                        data_simple["answer"].append(answer)
                        data_simple["contexts"].append(context)
                    st.session_state.simple_rag_df = pd.DataFrame(data_simple)
                    #st.write("Simple RAG Evaluation DataFrame:")
                    #st.dataframe(st.session_state.simple_rag_df)

                if "MultiQuery RAG" in selected_rag:
                    multiquery_rag_pipeline = MultiQueryPipeline(selected_llm, chunks)
                    data_multiquery = {
                        "question": [],
                        "answer": [],
                        "contexts": [],
                        "ground_truth": st.session_state.ground_truth,
                    }
                    for query in st.session_state.questions:
                        answer = multiquery_rag_pipeline.run(query)
                        context = multiquery_rag_pipeline.get_context(query)
                        logger.info(f"Multiquery RAG - Query: {query}, Contexts: {context}")
                        data_multiquery["question"].append(query)
                        data_multiquery["answer"].append(answer)
                        data_multiquery["contexts"].append(context)
                    st.session_state.multiquery_rag_df = pd.DataFrame(data_multiquery)
                    #st.write("MultiQuery RAG Evaluation DataFrame:")
                    #st.dataframe(st.session_state.multiquery_rag_df)

            if st.session_state.simple_rag_df is not None:
                st.write("Simple RAG Evaluation DataFrame:")
                st.dataframe(st.session_state.simple_rag_df)

            if st.session_state.multiquery_rag_df is not None:
                st.write("MultiQuery RAG Evaluation DataFrame:")
                st.dataframe(st.session_state.multiquery_rag_df)

    with st.expander("Evaluation"):
        selected_metrics = select_evaluation_metrics()

        if st.button("Generate Evaluation"):
            st.write("Selected LLM:", selected_llm)
            st.write("Selected RAG Pipeline(s):", selected_rag)
            st.write("Selected Dataset Generation:", selected_dataset)
            st.write("Selected Evaluation Metrics:", selected_metrics)

            metrics_map = {
                "context precision": context_precision,
                "context relevancy": context_relevancy,
                "answer relevancy": answer_relevancy,
            }
            selected_metrics_objs = [metrics_map[metric] for metric in selected_metrics]

            if st.session_state.simple_rag_df is not None:
                evaluator_simple = Evaluator(st.session_state.simple_rag_df)
                evaluation_result_simple = evaluator_simple.run_evaluation(
                    selected_metrics_objs
                )
                st.write("**Simple RAG Evaluation Results:**")
                st.dataframe(evaluation_result_simple)

            if st.session_state.multiquery_rag_df is not None:
                evaluator_multiquery = Evaluator(st.session_state.multiquery_rag_df)
                evaluation_result_multiquery = evaluator_multiquery.run_evaluation(
                    selected_metrics_objs
                )
                st.write("**MultiQuery RAG Evaluation Results:**")
                st.dataframe(evaluation_result_multiquery)

if __name__ == "__main__":
    main()
