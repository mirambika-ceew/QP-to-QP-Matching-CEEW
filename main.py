import streamlit as st

from utils.get_config import AppConfig
from utils.get_data_loader import DataLoader
from utils.setup_logger import setup_logger
from functions.get_summary_QP import render_summary_page
from utils.get_search_embeddings import EmbeddingGenerator
from functions.get_job2job_similarity import render_job_matching_page
from functions.get_job2job_similarity import render_job_description_matching


def main():

    # Load configuration
    config = AppConfig.from_yaml('config.yaml')
    logger = setup_logger(config.LOG_PATH)

    # Initialize data loader
    data_loader = DataLoader(config, logger)
    job_data, similarity_score = data_loader.load_data(config, logger)
    data_loader.job_data = job_data

    # Initialize embedding generator
    embedding_generator = EmbeddingGenerator()

    # Sidebar navigation
    page = st.sidebar.selectbox(
        "Select Page",
        ["Overview", "Job-Job Matching", "Summary", "Job Description Matching"]
    )

    _, col, _ = st.columns([1, 8, 1])

    with col:
        # Page routing
        if page == "Overview":
            st.subheader("Overview")
        elif page == "Job-Job Matching":
            render_job_matching_page(data_loader, similarity_score, logger)
        elif page == "Summary":
            render_summary_page(data_loader, job_data, config, logger)
        else:
            render_job_description_matching(data_loader, embedding_generator, logger)
