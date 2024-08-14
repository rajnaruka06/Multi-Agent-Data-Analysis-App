import streamlit as st
from dotenv import load_dotenv
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import ast
from collections import deque
import re
from io import StringIO

from workflows import DataAnalyticsWorkflow
from Agent_Helpers import get_sample_queries

if __name__ == "__main__":
    load_dotenv()

    st.title("Data Analyst Assistant")
    st.header("Some cool description here...")

    ## Init Resources
    workflow = DataAnalyticsWorkflow()
    sample_queries = get_sample_queries()

    ## Streamlit UI
    selected_sample = st.selectbox("Select a sample question:", sample_queries)
    user_query = st.text_area("Or Enter your Own question here: (leave blank to use selected option)", height=200, value=selected_sample)

    st.sidebar.title("Options")
    need_viz = st.sidebar.toggle("Generate Visualization", False)
    need_summary = st.sidebar.toggle("Generate Summary", False)
    show_sql = st.sidebar.toggle("Show SQL Query", False)
    show_viz_code = st.sidebar.toggle("Show Python Code for visualization", False)
    show_fetched_data = st.sidebar.toggle("Show Fetched Data", True)
    show_analyst_desc = st.sidebar.toggle("Show Analyst Description", False)

    if st.button("Get results"):
        user_query = selected_sample if user_query == "" else user_query

        with st.spinner("Querying Database..."):
            sql_query = workflow.generate_sql_query(user_query)

        if show_sql:
            st.write("---")
            st.subheader("Generated SQL Query:")
            st.write(sql_query)

        try:
            res = workflow.execute_sql_query(sql_query)
            workflow.hist.append(user_query)

        except Exception as e:
            res = f"Error: {e}. Please try refining your query."

        if show_fetched_data:
            st.write("---")
            st.subheader("Fetched Results:")
            st.write(res)

        if need_summary:
            if isinstance(res, str):
                summary = "Cannot generate summary for invalid data. Please try again."
            else:
                with st.spinner("Summarizing data..."):
                    summary = workflow.summarize_results(user_query, res)
            st.write("---")
            st.subheader("Summary:")
            st.write(summary)

        if need_viz:
            if isinstance(res, str):
                viz_code = "Cannot generate visualization for invalid data. Please try again."
            else:
                with st.spinner("Generating Visualization..."):
                    viz_code = workflow.generate_visualization(user_query, res)

                if show_viz_code:
                    st.write("---")
                    st.subheader("Visualization Code:")
                    st.code(viz_code, language='python')

                st.write("---")
                st.subheader("Visualization:")
                try:
                    fig = workflow.execute_viz_code(viz_code, res)
                    st.plotly_chart(fig)
                except Exception as e:
                    st.write(f"Error generating visualization: {e}")
