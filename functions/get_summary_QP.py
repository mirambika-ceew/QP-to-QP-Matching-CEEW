import streamlit as st
# pages/job_matching.py
import streamlit as st
import numpy as np
from pathlib import Path


def display_job_comparison(data_loader, job_data, sector1, job1, nsqf1, sector2, job2, nsqf2):
    technical_summary = data_loader.config.KEY_NAME["technical_summary"]
    behavioural_summary = data_loader.config.KEY_NAME["behavioural_summary"]
    overall_summary = data_loader.config.KEY_NAME["overall_summary"]

    # Extract details for Job 1
    job1_details = job_data[sector1][job1][nsqf1]
    job1_description = job1_details["job_description"]
    job1_technical = job1_details[technical_summary]
    job1_soft = job1_details[behavioural_summary]
    job1_overall = job1_details[overall_summary]

    # Extract details for Job 2
    job2_details = job_data[sector2][job2][nsqf2]
    job2_description = job2_details["job_description"]
    job2_technical = job2_details[technical_summary]
    job2_soft = job2_details[behavioural_summary]
    job2_overall = job2_details[overall_summary]

    # Layout with two columns
    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"{job1} (NSQF {nsqf1})")
        st.markdown(f"**Sector:** {sector1}")
        st.markdown(f"**Job Description:** {job1_description}")

    with col2:
        st.subheader(f"{job2} (NSQF {nsqf2})")
        st.markdown(f"**Sector:** {sector2}")
        st.markdown(f"**Job Description:** {job2_description}")
    st.markdown("</div>", unsafe_allow_html=True)
    st.divider()

    # Toggle switches
    with st.expander("Job Description", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Job Description:** {job1_description}")
        with col2:
            st.markdown(f"**Job Description:** {job2_description}")

    with st.expander("Technical Skills Summary", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Technical Skills Summary:** {job1_technical}")
        with col2:
            st.markdown(f"**Technical Skills Summary:** {job2_technical}")

    with st.expander("Behavioral Skills Summary", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Behavioral Skills Summary:** {job1_soft}")
        with col2:
            st.markdown(f"**Behavioral Skills Summary:** {job2_soft}")

    with st.expander("Overall Summary", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Overall Summary:** {job1_overall}")
        with col2:
            st.markdown(f"**Overall Summary:** {job2_overall}")

    st.divider()


def render_summary_page(data_loader, job_data, config, logger):
    st.title("Job Summary Comparison")

    col1, col2 = st.columns(2)

    with col1:
        sector1 = st.selectbox("Sector 1", data_loader.get_sectors(config))
        job1 = st.selectbox("Job 1", data_loader.get_jobs(sector1, job_data))

    with col2:
        sector2 = st.selectbox("Sector 2", data_loader.get_sectors(config))
        job2 = st.selectbox("Job 2", data_loader.get_jobs(sector2, job_data))

    job1_name, nsqf1 = job1.split("|")[0].strip(), job1.split("NSQF:")[1].strip()
    job2_name, nsqf2 = job2.split("|")[0].strip(), job2.split("NSQF:")[1].strip()

    display_job_comparison(data_loader, job_data, sector1, job1_name, nsqf1, sector2, job2_name, nsqf2)


