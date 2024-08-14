"""
Make agent with memory -- Done
Make agent with dashboard creation support -- Experimental. Let's ignore for now...
Make mongoDB agent
Make summarization Agent -- Working...
Make Routing between summarization and Dashboard Agent
"""
# db_agent_template = """Given an input question, just create a syntactically correct {dialect} query to run. 
# Do not include any CREATE, DELETE, UPDATE, or ALTER statements in your responses.
# You can use Common Table Expressions (CTEs) for data manipulation
# Use the following format:

# Question: Question here
# SQLQuery: SQL Query to run

# Only use the following tables:

# {table_info}.

# Question: {input}
# SQLQuery: 
# """

import streamlit as st
from dotenv import load_dotenv
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ast
from collections import deque
import re

from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

@st.cache_resource
def getDB():
    username = os.getenv('POSTGRES_USER')
    password = os.getenv('POSTGRES_PASSWORD')
    host = os.getenv('POSTGRES_HOST')
    port = os.getenv('POSTGRES_PORT')
    database = "chinook"
    postgresql_uri = f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{database}"

    db = SQLDatabase.from_uri(postgresql_uri)
    return db

@st.cache_resource
def getLLM(model_path = None):
    # if model_path is None:
    #     model_base_path = "C:\\Users\\rajna\\Desktop\\Studies\\Jupyter Notebook\\RAGS\\resources"
    #     model_name = "capybarahermes-2.5-mistral-7b.Q4_K_M.gguf"
    #     model_path = os.path.join(model_base_path, model_name)
    
    # llm = GPT4All(model=model_path, device="cuda", max_tokens=2048)
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2
    )
    return llm

@st.cache_resource
def get_sample_queries():
    sample_queries = [
        "Retrieve the album title and the artist name for each album."
        , 'Retrieve the first and last names of customers who have made purchases along with the total amount of their invoices.'
        , 'Retrieve the number of tracks in each album along with the album title and artist name. Sort the results by the number of tracks in descending order.'
        , 'Retrieve the artists name and number of tracks by each artist. Sort the results by the number of tracks in descending order. Only include artists with more than 10 tracks.'
        , "Retrieve the total sales (sum of invoice totals) for each artist, showing only the artist name and total sales. And give results in descending order of total sales."
        , "Retrieve the customer names (first and last) who have spent more than the average total of all invoices."
        , "Retrieve the names of the top 5 customers based on the total amount spent, along with the total amount they spent and the total number of invoices"
        , "Create a view to encapsulate the total sales for each customer and then use this view to retrieve customers who have spent more than $100"
    ]

    return sample_queries

@st.cache_resource
def init_history():
    return deque([], maxlen=3)

@st.cache_data
def get_table_definitions(table_names):
    table_definitions = dict()
    for table in table_names:
        table_definitions[table] = db.get_table_info([table]).strip().split('/*')[0]
    return table_definitions

def get_cols(sql_query):
    ## Find a SELECT then any number of any whitespace 
    ## (.+?) is a non-greedy match for any character except newline 
    ##   -> . is any character except newline
    ##   -> + is one or more of the preceding character
    ##   -> ? is a non-greedy match meaning it will match as few characters as possible
    ## () around this means we want to capture this group. so findall will return this group
    ## Ends with a FROM that comes after any number of any whitespace

    ## re.IGNORECASE is to ignore case sensitivity, re.DOTALL is to make . match newline characters as well.
    ## without re.DOTALL, . will not match newline characters 

    ## Essentially, we are capturing the columns between SELECT and FROM. any number of any whitespace after SELECT and before FROM will not be captured.

    select_regex = re.compile(r'SELECT\s+(.+?)\s+FROM', re.IGNORECASE | re.DOTALL)
    all_matches = select_regex.findall(sql_query)
    
    if not all_matches: return []
    final_select = all_matches[-1]
    columns = [col.strip() for col in final_select.split(',')]
    
    return columns

def execute_seaborn_code(seaborn_code, df):
    local_env = {'df': df, 'sns': sns, 'plt': plt, 'np': np}
    exec(seaborn_code, local_env)
    return local_env['fig']
    

if __name__ == "__main__":
    load_dotenv()

    st.title("PoliQ")
    st.header("Project Description goes here...")

    ## Init Resources
    db = getDB()
    llm = getLLM()
    query_runner = QuerySQLDataBaseTool(db = db)
    sample_queries = get_sample_queries()
    table_names = db.get_usable_table_names()
    table_definitions = get_table_definitions(table_names)
    hist = init_history()

    dba_agent_template = """Given an input question, just create a syntactically correct {dialect} query to run. 
    Do not include any CREATE, DELETE, UPDATE, or ALTER statements in your responses.
    Use Common Table Expressions (CTEs) for data manipulation instead of subqueries.
    Use the information from previous questions if they add context or relevant details to the current question. Focus primarily on the most recent and relevant questions. If the previous questions do not add any context, just focus on the current question.
    Use the following format:

    Previous Questions: List of previous questions
    Current Question: Curent User Question here
    SQLQuery: SQL Query to run

    Only use the following tables:

    {table_info}.

    Previous Questions: {previous_queries}
    Current Question: {input}
    SQLQuery: 
    """

    summary_agent_template = """You are a data analyst. Given a pandas DataFrame, summarize the data in a user-readable format. 
    Describe the key insights, trends, and any notable observations from the data. 
    Make sure to include statistics, comparisons, and any relevant details that provide a clear understanding of the data.
    
    DataFrame:
    {dataframe}
    Summary:
    """
    
    viz_agent_template = """
    You are a data visualization expert. Given a description of the desired visualization and a pandas DataFrame, generate just the Python code to create the visualization using Seaborn. 
    You can only use the following libraries: numpy (as np), pandas (as pd), seaborn (as sns), and matplotlib.pyplot (as plt).
    Do not include any import statements or definitions of the dataset. The dataset is available as df, and the modules are already imported.
    The generated code should just create the visualization figure object as fig and not return it.
    Ensure the code is complete and correct.
    
    Description: {description}
    DataFrame: {dataframe}
    Seaborn Code:
    """
    
    dba_agent_prompt = PromptTemplate.from_template(dba_agent_template)
    dba_chain = dba_agent_prompt | llm ## | execute_query

    summary_agent_prompt = PromptTemplate.from_template(summary_agent_template)
    summary_chain = summary_agent_prompt | llm

    viz_agent_prompt = PromptTemplate.from_template(viz_agent_template)
    viz_chain = viz_agent_prompt | llm


    ## Streamlit UI
    selected_sample = st.selectbox("Select a sample question:", sample_queries)
    user_query = st.text_area("Or Enter your Own question here: (leave blank to use selected option)", height=200, value = selected_sample)

    if st.button("Get results"):

        prev_queries = ''
        for idx, query in enumerate(hist):
            prev_queries += f"Question {idx+1}: {query}; "
            
        user_query = selected_sample if user_query == "" else user_query
        sql_query = dba_chain.invoke({
            "input": user_query
            , "dialect": db.dialect
            , "table_info": table_definitions
            , "previous_queries": prev_queries
        }).content.strip()

        # print('---')
        # print(prev_queries)

        sql_query = sql_query.replace('`', '')
        if sql_query.startswith('sql'): sql_query = sql_query[len('sql'):].strip()
        # if sql_query.startswith('SQLQuery:'): sql_query = sql_query[len('SQLQuery:'):].strip()
        if 'SQLQuery:' in sql_query: sql_query = sql_query.split('SQLQuery:')[1].strip()
        # print(sql_query)

        st.write("---")
        st.subheader("Generated SQL Query:")
        st.write(sql_query)

        st.write("---")
        st.subheader("Fetched Results:")
        try:
            if "CREATE" in sql_query or "DELETE" in sql_query or "UPDATE" in sql_query or "ALTER" in sql_query: 
                raise SyntaxError("Invalid SQL Query generated. Trying to run a DDL statement.")
            hist.append(user_query)

            res = query_runner.invoke(sql_query)
            res = res.replace('Decimal', '')
            res = ast.literal_eval(res)
            columns = get_cols(sql_query)
            if columns == []: columns = range(len(res[0]))
            res = pd.DataFrame.from_records(data = res, columns=columns)

        except SyntaxError:
            res = "Invalid SQL Query generated. Please try again. (maybe try rephrasing the question)"
        
        st.write(res)
        st.write("---")
        if st.checkbox("Summarize Results"):
            if res.startswith('Invalid SQL Query generated.'):
                summary = "Invalid SQL Query generated. Please try again. (maybe try rephrasing the question)"
            else:
                summary = summary_chain.invoke({
                    "dataframe": res.to_string()
                    }).content.strip()
                
            st.subheader("Summary:")
            st.write(summary)
        
        if isinstance(res, str):
            viz_code = "Invalid SQL Query generated. Please try again. (maybe try rephrasing the question)"
        else:
            viz_code = viz_chain.invoke({
                "description": "Visualize the data as a bar chart",
                "dataframe": res.to_string()
            }).content.strip()
            viz_code = viz_code.replace('`', '').strip()
            if viz_code.startswith('python'): viz_code = viz_code[len('python'):].strip()
                
            st.subheader("Visualization Code:")
            st.code(viz_code, language='python')

            st.write("---")
            fig = execute_seaborn_code(viz_code, res)
            st.pyplot(fig)