import streamlit as st
import sys, os
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)
import eda, classification

st.sidebar.title('Data Analysis')

page = st.sidebar.radio("Select a Page", ["EDA", "Classification"])


if page == 'EDA':
    eda.app()
elif page == 'Classification':
    classification.app()