import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingest import ingest_data
import os
from src import config

def ingestion_page():
    st.title("Data Ingestion")

    st.write("This page allows you to ingest your PDF documents into the vector store.")

    if st.button("Start Ingestion"):
        with st.spinner("Ingesting data... This may take a while."):
            ingest_data()
        st.success("Data ingestion complete!")

    st.write("### Current Data")
    if os.path.exists(config.DATA_PATH):
        files = os.listdir(config.DATA_PATH)
        if files:
            st.write("The following files are in your data directory:")
            for file in files:
                st.write(f"- {file}")
        else:
            st.warning("No files found in the data directory.")
    else:
        st.error("Data directory not found.")