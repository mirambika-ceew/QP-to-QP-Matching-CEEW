# data_loader.py
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import json
from pathlib import Path
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
import logging

from utils.utils import get_percentile_rank
from utils.get_config import AppConfig


class DataLoader:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.job_data = {}

    @st.cache_data
    def load_data(_self, config, _logger) -> Tuple[Dict, Dict]:
        try:
            with open(config.JOB_DATA_PATH, 'r') as f:
                job_data = json.load(f)
            similarity_data = {
                'technical_summary': pd.read_csv(config.SIMILARITY_DATA_PATH['technical_summary']),
                'behavioural_summary': pd.read_csv(config.SIMILARITY_DATA_PATH['behavioural_summary']),
                'overall_summary': pd.read_csv(config.SIMILARITY_DATA_PATH['overall_summary']),
            }
            return job_data, similarity_data
        except Exception as e:
            _logger.error(f"Error loading data: {e}")
            raise

    @st.cache_data
    def get_sectors(_self, config):
        return config.SUPPORTED_SECTORS

    @st.cache_data
    def get_jobs(_self, selected_sector, job_data):
        jobs = []
        for job, nsqfs in job_data[selected_sector].items():
            for nsqf in nsqfs.keys():
                jobs.append(f"{job} | NSQF:{nsqf}")
        return list(set(jobs))

    @st.cache_data
    def get_all_jobs(_self, job_data):
        jobs = []
        for sector, jobs_info in job_data.items():
            for job, nsqfs in jobs_info.items():
                for nsqf in nsqfs.keys():
                    jobs.append(f"{job} | NSQF:{nsqf}")
        return list(set(jobs))

    @st.cache_data
    def get_top_matches(_self, sectors, job, similarity_type, similarity_data, n_matches=5):
        similarity_scores = similarity_data[similarity_type]
        filtered_similarity_scores = similarity_scores.loc[(similarity_scores['Sector2'].isin(sectors)) &
                                                           (similarity_scores['Job1'] == job)]
        reference_job, reference_sector, reference_nsqf = 'Job2', 'Sector2', 'NSQF2'
        if len(filtered_similarity_scores) == 0:
            filtered_similarity_scores = similarity_scores.loc[(similarity_scores['Sector1'].isin(sectors)) &
                                                           (similarity_scores['Job2'] == job)]
            reference_job, reference_sector, reference_nsqf = 'Job1', 'Sector1', 'NSQF1'

        filtered_similarity_scores = filtered_similarity_scores[['similarity_score',
                                                                 reference_job, reference_sector, reference_nsqf]]
        filtered_similarity_scores = filtered_similarity_scores.sort_values(by=['similarity_score'], ascending=False)
        filtered_similarity_scores = filtered_similarity_scores.iloc[0: n_matches]
        filtered_similarity_scores['similarity_score'] = filtered_similarity_scores['similarity_score'].apply(lambda x : np.round(x, 2))
        return filtered_similarity_scores
    @st.cache_data
    def get_job_pair_similarity(_self, sector1, job1, sector2, job2, similarity_type, similarity_data):
        similarity_scores = similarity_data[similarity_type]
        similarity_scores['NSQF1'] = similarity_scores['NSQF1'].apply(lambda x: str(np.round(x, 1)))
        similarity_scores['NSQF2'] = similarity_scores['NSQF2'].apply(lambda x: str(np.round(x, 1)))
        job1_name, nsqf1 = job1.split("|")[0].strip(), job1.split("NSQF:")[1].strip()
        job2_name, nsqf2 = job2.split("|")[0].strip(), job2.split("NSQF:")[1].strip()

        filtered_similarity_scores = similarity_scores.loc[(similarity_scores['Job1'] == job1_name) &
                                                           (similarity_scores['Job2'] == job2_name) &
                                                           (similarity_scores['NSQF1'] == nsqf1) &
                                                           (similarity_scores['NSQF2'] == nsqf2)]

        if len(filtered_similarity_scores) == 0:
            filtered_similarity_scores = similarity_scores.loc[(similarity_scores['Job1'] == job2_name) &
                                                               (similarity_scores['Job2'] == job1_name) &
                                                               (similarity_scores['NSQF1'] == nsqf2) &
                                                               (similarity_scores['NSQF2'] == nsqf1)]

        similarity = filtered_similarity_scores['similarity_score'].values[0]
        return similarity

    @st.cache_data
    def get_percentile_score_info(_self, similarity, all_similarity_scores):
        # all_similarity_scores = self.similarity_data[similarity_type]['similarity_score'].values.squeeze()
        percentile_info = {
            'lower_percentile': 5,
            'lower_value': np.percentile(all_similarity_scores, 5),
            'upper_percentile': 95,
            'upper_value': np.percentile(all_similarity_scores, 95),
            'score_percentile': get_percentile_rank(similarity, list(all_similarity_scores))
        }
        return percentile_info

    @st.cache_data
    def get_cross_sector_similarities(_self, sector1, sector2, similarity_type, similarity_data, n_pairs=10):
        similarity_scores = similarity_data[similarity_type]
        filtered_similarity_scores = similarity_scores.loc[(similarity_scores['Sector1'] == sector1) &
                                                           (similarity_scores['Sector2'] == sector2)]
        if len(filtered_similarity_scores) == 0:
            filtered_similarity_scores = similarity_scores.loc[(similarity_scores['Sector2'] == sector1) &
                                                           (similarity_scores['Sector1'] == sector2)]
        filtered_similarity_scores = filtered_similarity_scores.sort_values(by=['similarity_score'], ascending=False)
        filtered_similarity_scores['similarity_score'] = filtered_similarity_scores['similarity_score'].apply(lambda x: np.round(x, 2))
        filtered_similarity_scores = filtered_similarity_scores.iloc[0: n_pairs+1]
        return filtered_similarity_scores
    @st.cache_data
    def get_sector_similarity_matrix(_self, similarity_data, similarity_type):
        similarity_scores = similarity_data[similarity_type]
        text_average_similarity = similarity_scores.groupby(['Sector1', 'Sector2'])[
            'similarity_score'].mean().reset_index()
        text_average_similarity = text_average_similarity.sort_values(by='similarity_score', ascending=False).reset_index(
            drop=True)
        pivot_df = text_average_similarity.pivot(index='Sector1', columns='Sector2', values='similarity_score')
        return pivot_df

