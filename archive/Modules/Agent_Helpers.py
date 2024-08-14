from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_community.utilities import SQLDatabase
import os
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import ast
import re



def init_history(lim = 3) -> list:
    from collections import deque
    return deque([], maxlen=lim)

def get_table_definitions(db: SQLDatabase) -> dict:
    ## Used only the table definitinos first because of context lentgh limit

    # table_names = db.get_usable_table_names()
    # table_definitions = dict()
    # for table in table_names:
    #     table_definitions[table] = db.get_table_info([table]).strip().split('/*')[0]
    # return table_definitions
    return db.get_table_info()

def get_sample_queries():
    sample_queries = [
        "Retrieve the album title and the artist name for each album."
        , 'Retrieve the first and last names of customers who have made purchases along with the total amount of their invoices.'
        , 'Retrieve the number of tracks in each album along with the album title and artist name. Sort the results by the number of tracks in descending order.'
        , 'Retrieve the artists name and number of tracks by each artist. Sort the results by the number of tracks in descending order. Only include artists with more than 10 tracks.'
        , "Retrieve the total sales (sum of invoice totals) for each artist, showing only the artist name and total sales. And give results in descending order of total sales."
        , "Retrieve the customer names (first and last) who have spent more than the average total of all invoices."
        , "Retrieve the names of the top 5 customers based on the total amount spent, along with the total amount they spent and the total number of invoices"
        , "Retrieve the customer details who have spent more than $30"
    ]
    return sample_queries

def execute_viz_code(viz_code: str, df: pd.DataFrame) -> go.Figure:
    local_env = {'df': df, 'np': np, 'pd': pd, 'go': go}
    exec(viz_code, local_env)
    return local_env['fig']

def clean_sql_query(sql_query: str) -> str:
    sql_query = sql_query.replace('`', '').strip()
    if sql_query.startswith('sql'):
        sql_query = sql_query[len('sql'):].strip()
    if 'SQLQuery:' in sql_query:
        sql_query = sql_query.split('SQLQuery:')[1].strip()
    return sql_query

class DDLCommandException(Exception):
    "Raised when SQL contains DDL commands (CREATE, DELETE, UPDATE, ALTER)"
    pass

class NoDataFoundException(Exception):
    "Raised when no data is fetched from the database"
    pass

class DBLoader:
    def __init__(self):
        load_dotenv()
        self.username = os.getenv('POSTGRES_USER')
        self.password = os.getenv('POSTGRES_PASSWORD')
        self.host = os.getenv('POSTGRES_HOST')
        self.port = os.getenv('POSTGRES_PORT')
        self.database = os.getenv('POSTGRES_DB_NAME')
        self.postgresql_uri = f"postgresql+psycopg2://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"

    def load_db(self) -> SQLDatabase:
        db = SQLDatabase.from_uri(self.postgresql_uri)
        return db

class SQLCoder:
    def __init__(self, db: SQLDatabase):
        self.db = db
        self.query_runner = QuerySQLDataBaseTool(db=db)

    def execute_query(self, query: str) -> pd.DataFrame:
        try:
            res = self.query_runner.invoke(query)
            res = res.replace('Decimal', '')
            if res == '':
                raise NoDataFoundException
            res = ast.literal_eval(res)
            columns = self.get_cols(query)
            if columns == []: columns = range(len(res[0]))
            res = pd.DataFrame.from_records(data=res, columns=columns)
            return res
        except Exception as e:
            raise e

    def get_cols(self, sql_query: str) -> list:
        select_regex = re.compile(r'SELECT\s+(.+?)\s+FROM', re.IGNORECASE | re.DOTALL)
        all_matches = select_regex.findall(sql_query)
        if not all_matches: return []
        final_select = all_matches[-1]
        columns = [col.strip() for col in final_select.split(',')]
        return columns
