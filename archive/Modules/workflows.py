from CustomAgents import ResponseSummarizer, VisualizationAgent, AnalystAgent, SQLExpert
from Agent_Helpers import DBLoader, SQLCoder, init_history, execute_viz_code, DDLCommandException, NoDataFoundException, get_table_definitions, clean_sql_query
from langchain_openai import ChatOpenAI
from io import StringIO
import os
from datetime import datetime

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataAnalyticsWorkflow:
    def __init__(self):
        self.db_loader = DBLoader()
        self.db = self.db_loader.load_db()
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2
        )
        self.sql_coder = SQLCoder(self.db)
        self.response_summarizer = ResponseSummarizer(self.llm)
        self.visualization_agent = VisualizationAgent(self.llm)
        self.analyst_agent = AnalystAgent(self.llm)
        self.query_generator = SQLExpert(self.llm)
        self.hist = init_history()

    def generate_sql_query(self, user_query):
        prev_queries = '; '.join([f"Question {idx+1}: {query}" for idx, query in enumerate(self.hist)])
        sql_query = self.query_generator.generate_query(
            user_query,
            self.db.dialect,
            get_table_definitions(self.db),
            prev_queries
        )
        return clean_sql_query(sql_query)

    def execute_sql_query(self, sql_query):
        if "CREATE" in sql_query or "DELETE" in sql_query or "UPDATE" in sql_query or "ALTER" in sql_query:
            raise DDLCommandException
        return self.sql_coder.execute_query(sql_query)

    def summarize_results(self, user_query, res):
        if isinstance(res, str):
            return "Cannot generate summary for invalid data. Please try again."
        return self.response_summarizer.summarize(user_query, res)

    def generate_visualization(self, user_query, res):
        if isinstance(res, str):
            return "Cannot generate visualization for invalid data. Please try again."
        head = res.head().to_dict()
        buffer = StringIO()
        res.info(buf=buffer)
        info = buffer.getvalue()
        data_desc = res.describe().to_string()

        viz_desc = self.analyst_agent.generate_viz_description(user_query, head, info, data_desc)
        viz_code = self.visualization_agent.generate_viz_code(viz_desc, res)
        return viz_code

    def save_visualization(self, viz_code, res):
        try:
            fig = execute_viz_code(viz_code, res)
            directory = "Viz History"
            if not os.path.exists(directory):
                os.makedirs(directory)
            filename = datetime.now().strftime("%Y%m%d_%H%M%S") + ".html"
            filepath = os.path.join(directory, filename)
            fig.write_html(filepath)
            logger.info("Visualization saved as %s", filepath)
        except Exception as e:
            logger.error("Error generating visualization: %s", e)

    def run_workflow(self, user_query):
        try:
            sql_query = self.generate_sql_query(user_query)
            logger.info("Generated SQL Query:\n%s", sql_query)

            res = self.execute_sql_query(sql_query)
            logger.info("Fetched Results:\n%s", res)

            summary = self.summarize_results(user_query, res)

            viz_code = self.generate_visualization(user_query, res)
            logger.info("Visualization Code:\n%s", viz_code)

            self.save_visualization(viz_code, res)
            self.hist.append(user_query)

        except DDLCommandException:
            logger.error("Invalid SQL Query generated. DDL commands are not allowed. Please try again.")
        except SyntaxError:
            logger.error("Invalid SQL Query generated. Please try again.")
        except NoDataFoundException:
            logger.error("No data found for the query. Please try refining your query")
        except Exception as e:
            logger.error("Error: %s. Please try refining your query.", e)
