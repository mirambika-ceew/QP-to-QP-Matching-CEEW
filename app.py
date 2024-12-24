import streamlit as st

from main import main

st.markdown("""
    <style>
    /* Targeting the content inside the st.data_editor cells */
    .stDataFrame {
        font-size: 40px !important;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
    /* For st.write text */
    div[data-testid="stText"], div[data-testid="stMarkdown"] {
        font-size: 35px !important;
    }

    /* For selectbox */
    div[data-baseweb="select"] > label {
        font-size: 35px !important;
    }

    /* For multiselect */
    div[data-baseweb="multi-select"] > label {
        font-size: 35px !important;
    }

    /* For number input */
    div[data-testid="stNumberInput"] > label {
        font-size: 35px !important;
    }

    /* For tabs */
    button[data-baseweb="tab"] {
        font-size: 40px !important;
    }

    /* For radio buttons */
    div[data-testid="stRadio"] > label {
        font-size: 35px !important;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
    div.element-container {
        padding-top: 0rem;
        padding-bottom: 0rem;
    }
    </style>
""", unsafe_allow_html=True)

main()