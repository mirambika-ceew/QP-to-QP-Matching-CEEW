import logging
import numpy as np
import pandas as pd
import streamlit as st
from utils.get_data_loader import DataLoader
from utils.get_search_embeddings import EmbeddingGenerator
from utils.utils import add_colorscale
from utils.utils import highlight_rows

import json
import logging
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from typing import List


def get_percentile_rank(number: float, number_list: list) -> float:
    # Returns what percentage of numbers are below this number
    return (np.searchsorted(np.sort(number_list), number) / len(number_list)) * 100

def change_label_style(label, font_size='12px', font_color='black', font_family='sans-serif'):
    html = f"""
    <script>
        var elems = window.parent.document.querySelectorAll('p');
        var elem = Array.from(elems).find(x => x.innerText == '{label}');
        elem.style.fontSize = '{font_size}';
        elem.style.color = '{font_color}';
        elem.style.fontFamily = '{font_family}';
    </script>
    """
    st.components.v1.html(html)

def render_top_matching_jobs(data_loader: DataLoader, similarity_scores: dict, comparison_type: str,
                             logger: logging.Logger):
    """Renders the interface for finding top 5 matching jobs within a sector."""
    try:
        st.subheader(
            f"Identifies the top matching jobs from a sector for a given job, based on {comparison_type.split('_')[0]} skills")

        config = data_loader.config
        job_data = data_loader.job_data

        col1, col2 = st.columns(2)

        with col1:
            # User inputs
            st.divider()
            st.write("Select Current Sector: ")
            selected_sector1 = st.selectbox(
                "",
                data_loader.get_sectors(config)
            )

            st.divider()
            st.write("Select Current Job: ")
            selected_job = st.selectbox(
                "",
                data_loader.get_jobs(selected_sector1, job_data)
            )

        with col2:
            label = ""
            st.divider()
            st.write("Select reference sectors: ")
            selected_sectors = st.multiselect(label, data_loader.get_sectors(config))

            st.divider()
            st.write("Enter Top number of Jobs to display: ")
            n_matches = st.number_input(
                "",
                min_value=0,
                max_value=100,
                value=10,
                step=5
            )

        if st.button("Find Top Matches"):
            # Get similarity scores
            selected_job = selected_job.split("|")[0].strip()

            similarity_type = comparison_type.lower()
            matches = data_loader.get_top_matches(
                sectors=selected_sectors,
                job=selected_job,
                similarity_type=similarity_type,
                similarity_data=similarity_scores,
                n_matches=n_matches
            )

            # Display the dataframe of top matches
            _, col, _ = st.columns([1, 8, 1])
            matches = matches.reset_index().drop(['index'], axis=1)
            all_similarity_scores = similarity_scores[similarity_type]['similarity_score'].values.squeeze()
            percentile_info = data_loader.get_percentile_score_info(0, all_similarity_scores)
            subset = similarity_scores[similarity_type]
            cross_similarity_scores = subset.loc[subset['Sector1'] != subset['Sector2']]['similarity_score'].values.squeeze()
            cross_sector_percentile_info = data_loader.get_percentile_score_info(0, cross_similarity_scores)

            st.info(f"""
                Similarity Score Distribution across all jobs:
                - Lower {percentile_info['lower_percentile']}th percentile: {percentile_info['lower_value']:.2f}
                - Upper {percentile_info['upper_percentile']}th percentile: {percentile_info['upper_value']:.2f}
            """)

            st.info(f"""
                Similarity Score Distribution across jobs in different sectors:
                - Lower {cross_sector_percentile_info['lower_percentile']}th percentile: {cross_sector_percentile_info['lower_value']:.2f}
                - Upper {cross_sector_percentile_info['upper_percentile']}th percentile: {cross_sector_percentile_info['upper_value']:.2f}
            """)


            st.data_editor(
                matches,
                column_config={
                    "similarity_score": st.column_config.ProgressColumn(
                        "similarity score",
                        format="%.2f",
                        min_value=0,
                        max_value=1,
                    ),
                },
                hide_index=True,
                use_container_width=True
            )

            st.download_button(
                label="Download as CSV",
                data=matches.to_csv(index=False),
                file_name=f"top_matches_{selected_job}.csv",
                mime="text/csv"
            )

    except Exception as e:
        logger.error(f"Error in render_top_matching_jobs: {str(e)}")
        st.error("An error occurred while processing your request.")


def render_job_pair_similarity(data_loader: DataLoader, similarity_scores: dict, comparison_type: str,
                               logger: logging.Logger):
    """Renders the interface for comparing similarity between two specific jobs."""
    try:
        st.subheader(f"Calculates the cosine similarity for two jobs based on {comparison_type.split('_')[0]} skills")
        st.divider()

        col1, col2 = st.columns(2)
        config = data_loader.config
        job_data = data_loader.job_data
        all_sectors = data_loader.get_sectors(config)
        with col1:
            sector1 = st.selectbox("Select Sector 1", all_sectors, key="sector1")
            job1 = st.selectbox("Select Job 1", data_loader.get_jobs(sector1, job_data), key="job1")
        with col2:
            sector2 = st.selectbox("Select Sector 2", all_sectors, key="sector2")
            job2 = st.selectbox("Select Job 2", data_loader.get_jobs(sector2, job_data), key="job2")

        if job1 == job2:
            st.error("Please enter different jobs for valid comparison")

        elif st.button("Calculate Similarity"):
            similarity_type = comparison_type.lower()

            # Get similarity score and percentile information
            score = data_loader.get_job_pair_similarity(
                sector1=sector1,
                job1=job1,
                sector2=sector2,
                job2=job2,
                similarity_type=similarity_type,
                similarity_data=similarity_scores
            )
            all_similarity_scores = similarity_scores[similarity_type]['similarity_score'].values.squeeze()
            percentile_info = data_loader.get_percentile_score_info(score, all_similarity_scores)

            subset = similarity_scores[similarity_type]
            cross_similarity_scores = subset.loc[subset['Sector1'] != subset['Sector2']]['similarity_score'].values.squeeze()
            cross_sector_percentile_info = data_loader.get_percentile_score_info(score, cross_similarity_scores)

            # Create gauge chart for similarity score
            st.subheader(f"Similarity Score between : \n | {job1} | and \n | {job2} |")
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score,  # Convert to percentage
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 1]},
                    'steps': [
                        {'range': [0, 33], 'color': "lightgray"},
                        {'range': [33, 66], 'color': "gray"},
                        {'range': [66, 100], 'color': "darkgray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': score
                    }
                }
            ))

            st.plotly_chart(fig)

            # Display percentile information
            st.markdown("---")
            with st.container():
                st.info(f"""
                    Similarity Score Distribution across all jobs::
                    - Lower {percentile_info['lower_percentile']}th percentile: {percentile_info['lower_value']:.2f}
                    - Upper {percentile_info['upper_percentile']}th percentile: {percentile_info['upper_value']:.2f}
                    - Your score is in the {percentile_info['score_percentile']:.0f}th percentile
                """)
                st.info(f"""
                    Similarity Score Distribution across jobs in different sectors::
                    - Lower {cross_sector_percentile_info['lower_percentile']}th percentile: {cross_sector_percentile_info['lower_value']:.2f}
                    - Upper {cross_sector_percentile_info['upper_percentile']}th percentile: {cross_sector_percentile_info['upper_value']:.2f}
                    - Your score is in the {cross_sector_percentile_info['score_percentile']:.0f}th percentile
                """)

            # Download options
            col1, col2 = st.columns(2)
            with col1:
                # Download as JSON
                json_data = json.dumps({
                    'job1': job1,
                    'sector1': sector1,
                    'job2': job2,
                    'sector2': sector2,
                    'similarity_score': score,
                    'percentile_info': percentile_info})

                st.download_button(
                    label="Download as JSON",
                    data=json_data,
                    file_name="similarity_analysis.json",
                    mime="application/json"
                )

    except Exception as e:
        logger.error(f"Error in render_job_pair_similarity: {str(e)}")
        st.error("An error occurred while processing your request.")


def render_cross_sector_analysis(data_loader: DataLoader, similarity_scores: dict, comparison_type: str,
                                 logger: logging.Logger):
    """Renders the interface for analyzing top similar jobs between two sectors."""

    try:
        st.subheader('Cross Sector Analysis')
        st.divider()

        # User inputs
        config = data_loader.config
        job_data = data_loader.job_data

        col1, col2 = st.columns(2)
        all_sectors = data_loader.get_sectors(config)
        with col1:
            sector1 = st.selectbox("Select First Sector", all_sectors, key="cross_sector1")
        with col2:
            sector2 = st.selectbox("Select Second Sector", all_sectors, key="cross_sector2")
        n_pairs = st.number_input(
            "Enter Top number of Jobs to display:",
            min_value=0,
            max_value=100,
            value=5,
            step=5,
            key="number_jobs_cross_sector")

        if st.button("Find Similar Jobs"):
            similarity_type = comparison_type.lower()

            # Get top 10 similar job pairs
            similar_jobs = data_loader.get_cross_sector_similarities(
                sector1=sector1,
                sector2=sector2,
                similarity_type=similarity_type,
                similarity_data=similarity_scores,
                n_pairs=n_pairs
            )
            similar_jobs = similar_jobs.reset_index().drop(['index'], axis=1)
            similar_jobs = similar_jobs[['similarity_score', 'Job1', 'Job2', 'Sector1', 'Sector2', 'NSQF1', 'NSQF2']]
            similar_jobs['similarity_score'] = similar_jobs['similarity_score'].apply(lambda x : np.round(x, 3))
            similar_jobs['similarity_score'] = similar_jobs['similarity_score'].round(3)

            st.markdown("""
                <style>
                /* Center the text in header cells */
                .stDataFrame th {
                    font-size: 20px;
                    text-align: center;
                }
                /* Center the text in data cells */
                .stDataFrame td {
                    font-size: 18px;
                    text-align: center;
                }
                </style>
            """, unsafe_allow_html=True)

            similar_jobs = similar_jobs.style.format({'similarity_score': '{:.2f}'}) \

            st.dataframe(similar_jobs)

            # st.data_editor(
            #     similar_jobs,
            #     hide_index=True,
            #     use_container_width=True
            # )

    except Exception as e:
        logger.error(f"Error in render_cross_sector_analysis: {str(e)}")
        st.error("An error occurred while processing your request.")


def render_sector_similarity_matrix(data_loader: DataLoader, similarity_scores: dict, comparison_type: str,
                                    logger: logging.Logger):
    """Renders the sector-wise similarity matrix as a heatmap."""
    try:
        st.subheader('Average Similarity scores between Sectors')
        st.divider()
        # Get sector similarity matrix
        similarity_type = comparison_type.lower()
        similarity_matrix = data_loader.get_sector_similarity_matrix(similarity_scores, similarity_type)

        fig, ax = plt.subplots()
        sns.heatmap(similarity_matrix, annot=True, cmap='coolwarm', ax=ax, cbar_kws={'label': 'Similarity Score'})
        st.pyplot(fig)
        st.markdown("---")

        # Download options
        col1, col2 = st.columns([4,4])
        with col1:
            # Download as CSV
            csv_data = similarity_matrix.to_csv()
            st.download_button(
                label="Download as CSV",
                data=csv_data,
                file_name="sector_similarity_matrix.csv",
                mime="text/csv"
            )

    except Exception as e:
        logger.error(f"Error in render_sector_similarity_matrix: {str(e)}")
        st.error("An error occurred while processing your request.")


def find_matching_jobs(
        job_description: str,
        selected_sectors: List[str],
        comparison_type: str,
        data_loader: DataLoader,
        embedding_generator: EmbeddingGenerator,
        top_k: int = 10,
        similarity_threshold: float = 0.3,
) -> None:
    """
    Find and display jobs matching the input job description using vector similarity.

    Args:
        job_description: Input job description text
        selected_sectors: List of sectors to search in
        comparison_type: Type of comparison (Technical/Soft/Combined)
        data_loader: DataLoader instance
        embedding_generator: EmbeddingGenerator instance
        top_k: Number of top matches to display
        similarity_threshold: Minimum similarity score to consider
        logger: Logger object
    """
    try:
        # Show progress bar for embedding generation
        with st.spinner("Generating embeddings for job description..."):
            query_embedding = embedding_generator.generate_embedding(job_description)

        # Get relevant summaries and compute similarities
        matches = []
        comparison_type = comparison_type.lower()

        # Progress bar for similarity computation
        progress_bar = st.progress(0)
        job_count = sum(len(data_loader.job_data[sector]) for sector in selected_sectors)
        processed_jobs = 0

        for sector in selected_sectors:
            sector_data = data_loader.job_data[sector]

            for job_name, job_info in sector_data.items():
                for exp_level, details in job_info.items():
                    # Select appropriate summary based on comparison type
                    if comparison_type == 'technical':
                        summary = details['technical_summary']
                    elif comparison_type == 'soft':
                        summary = details['soft_skills_summary']
                    else:  # combined
                        summary = details['combined_summary']

                    # Get pre-computed embedding or compute if needed
                    job_embedding = data_loader.get_embedding(
                        sector, job_name, exp_level, comparison_type
                    )

                    # Compute similarity
                    similarity = embedding_generator.compute_similarity(
                        query_embedding, job_embedding
                    )

                    if similarity >= similarity_threshold:
                        matches.append({
                            'sector': sector,
                            'job_name': job_name,
                            'experience_level': exp_level,
                            'similarity_score': similarity,
                            'summary': summary
                        })

                # Update progress
                processed_jobs += 1
                progress_bar.progress(processed_jobs / job_count)

        # Clear progress bar
        progress_bar.empty()

        if not matches:
            st.warning("No matching jobs found above the similarity threshold.")
            return

        # Sort matches by similarity score and take top k
        matches.sort(key=lambda x: x['similarity_score'], reverse=True)
        top_matches = matches[:top_k]

        # Create DataFrame for visualization
        df_matches = pd.DataFrame(top_matches)

        # Create visualization
        fig = px.bar(
            df_matches,
            x='similarity_score',
            y='job_name',
            color='sector',
            orientation='h',
            title=f'Top {top_k} Matching Jobs',
            labels={
                'similarity_score': 'Similarity Score',
                'job_name': 'Job Name',
                'sector': 'Sector'
            },
            hover_data=['experience_level']
        )

        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            showlegend=True,
            height=600
        )

        # Display plot
        st.plotly_chart(fig)

        # Display detailed results in expandable sections
        st.subheader("Detailed Match Information")
        for idx, match in enumerate(top_matches, 1):
            with st.expander(
                    f"{idx}. {match['job_name']} ({match['sector']}) - "
                    f"Similarity: {match['similarity_score']:.2f}"
            ):
                st.write("**Experience Level:**", match['experience_level'])
                st.write("**Summary:**", match['summary'])

                # Add view full details button
                if st.button(f"View Full Details #{idx}"):
                    full_details = data_loader.get_job_details(
                        match['sector'],
                        match['job_name'],
                        match['experience_level']
                    )
                    st.json(full_details)

        # Download options
        col1, col2 = st.columns(2)
        with col1:
            # Download as CSV
            csv_data = df_matches.to_csv(index=False)
            st.download_button(
                label="Download Results as CSV",
                data=csv_data,
                file_name="job_matches.csv",
                mime="text/csv"
            )

        with col2:
            # Download as PNG
            img_bytes = fig.to_image(format="png")
            st.download_button(
                label="Download Visualization as PNG",
                data=img_bytes,
                file_name="job_matches.png",
                mime="image/png"
            )

    except Exception as e:
        st.error(f"An error occurred while processing your request: {str(e)}")
        # logger.error(f"Error in find_matching_jobs: {str(e)}", exc_info=True)


def render_job_matching_page(data_loader, similarity_scores, logger):
    st.title("Job to Job Matching using Qualification Packs")
    st.divider()

    st.write("Select Criteria for comparison")
    comparison_type = st.radio("",
                               ["Only Technical skills", "Only Soft skills", "Overall skills"])

    if "technical" in comparison_type.lower():
        comparison_type = "technical_summary"
    elif "soft" in comparison_type.lower():
        comparison_type = "behavioural_summary"
    elif "overall" in comparison_type.lower():
        comparison_type = "overall_summary"

    tab1, tab2, tab3, tab4 = st.tabs([
        "Job Pair Similarity",
        "Top Matching Jobs",
        "Sector Similarity Matrix",
        "Cross-Sector Analysis",
    ])

    with tab1:
        render_job_pair_similarity(data_loader, similarity_scores, comparison_type, logger)

    with tab2:
        render_top_matching_jobs(data_loader, similarity_scores, comparison_type, logger)

    with tab3:
        render_sector_similarity_matrix(data_loader, similarity_scores, comparison_type, logger)

    with tab4:
        render_cross_sector_analysis(data_loader, similarity_scores, comparison_type, logger)


# pages/job_description_matching.py
def render_job_description_matching(data_loader, embedding_generator, logger):
    st.title("Job Description Matching")

    job_description = st.text_area("Enter Job Description")
    selected_sectors = st.multiselect("Select Sectors", data_loader.get_sectors(), default=data_loader.get_sectors())
    comparison_type = st.radio("Comparison Type", ["Technical", "Soft", "Combined"])

    if st.button("Find Matches"):
        find_matching_jobs(job_description, selected_sectors, comparison_type, data_loader, embedding_generator)
