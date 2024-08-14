## Thoughts:
## - Move prmots and chains to init methods of the classes. This will optimize the code for redundant chain initializations.
## - But result in increased startup time as all chains will be initialized at once. Also, more memory usage.
## - Also difficult to manage the chains and prompts in the code.

from langchain_core.prompts import PromptTemplate

## Using CRAG idea to Correct the generated SQL query if needed. I rarely see CRAG being helpful here, there is no need for iterative refinement.
## only see CRAG used sometimes in Select * from table_name queries.
class SQLExpert:
    def __init__(self, llm):
        self.llm = llm

    def generate_query(self, user_query: str, dialect: str, table_info: str, previous_queries: str) -> str:
        dba_agent_template = """Given an input question, just create a syntactically correct {dialect} query to run.
        Do not include any CREATE, DELETE, UPDATE, or ALTER statements in your responses.
        Use Common Table Expressions (CTEs) for data manipulation instead of subqueries.
        Never use * in the SELECT statement. Always specify the columns you want to retrieve. Even if you are querying from Common Table Expressions.
        Use group by instead of distinct where applicable.
        Do not include any LIMIT, or OFFSET clauses in your responses unless the user query requires it.
        Use the information from previous questions if they add context or relevant details to the current question. Focus primarily on the most recent and relevant questions. If the previous questions do not add any context, just focus on the current question.
        Use the following format:

        Previous Questions: List of previous questions
        Current Question: Current User Question here
        SQLQuery: SQL Query to run

        Only use the following tables:

        {table_info}.

        Previous Questions: {previous_queries}
        Current Question: {user_query}
        SQLQuery:
        """
        dba_agent_prompt = PromptTemplate.from_template(dba_agent_template)
        dba_chain = dba_agent_prompt | self.llm
        sql_query = dba_chain.invoke({
            "user_query": user_query,
            "dialect": dialect,
            "table_info": table_info,
            "previous_queries": previous_queries
        }).content.strip()


        sql_query = self.correct_query(sql_query, dialect, table_info, user_query, previous_queries)

        return sql_query

    def correct_query(self, sql_query: str, dialect: str, table_info: str, user_query: str, previous_queries: str) -> str:
        corrective_template = """You are a SQL expert. Given a generated SQL query, evaluate it to ensure it adheres to the following criteria:
        - Do not include any CREATE, DELETE, UPDATE, or ALTER statements in your responses.
        - Use Common Table Expressions (CTEs) for data manipulation instead of subqueries.
        - Never use * in the SELECT statement. Always specify the columns you want to retrieve. Even if you are querying from Common Table Expressions.
        - Use group by instead of distinct where applicable.
        - Do not include any LIMIT, or OFFSET clauses in your responses unless the user query requires it.
        - Use the information from previous questions if they add context or relevant details to the current question. Focus primarily on the most recent and relevant questions. If the previous questions do not add any context, just focus on the current question.
        - Check whether this is the best way to answer the user query or if there could be a better way (by using different aggregations or different tables, etc.).

        If the query is already good, your response should just be one word: "All good". Otherwise, provide feedback and suggest improvements. Do not include an improved query.

        User Query: {user_query}
        Previous Questions: {previous_queries}
        Generated SQL Query: {sql_query}
        Dialect: {dialect}
        Table Info: {table_info}
        Correction:
        """
        corrective_prompt = PromptTemplate.from_template(corrective_template)
        corrective_chain = corrective_prompt | self.llm
        correction_result = corrective_chain.invoke({
            "user_query": user_query,
            "sql_query": sql_query,
            "dialect": dialect,
            "table_info": table_info,
            "previous_queries": previous_queries
        }).content.strip()

        # print("\n--- Generated SQL Query ---\n", sql_query)
        # print("\n--- Correction ---\n", correction_result)
        if correction_result.lower().strip() != "all good":
            sql_query = self.adjust_query(sql_query, dialect, table_info, user_query, correction_result, previous_queries)
            # print("\n--- Adjusted SQL Query ---\n", sql_query)

        return sql_query

    def adjust_query(self, sql_query: str, dialect: str, table_info: str, user_query: str, correction: str, previous_queries: str) -> str:
        adjustment_template = """You are a SQL expert. Given a generated SQL query, correction feedback, dialect, table information, and user query, adjust the query to make it more accurate and better answer the user query.
        Your response should just be a syntactically correct {dialect} query to run.

        User Query: {user_query}
        Previous Questions: {previous_queries}
        Generated SQL Query: {sql_query}
        Dialect: {dialect}
        Table Info: {table_info}
        Correction Feedback: {correction}
        Adjusted SQL Query:
        """
        adjustment_prompt = PromptTemplate.from_template(adjustment_template)
        adjustment_chain = adjustment_prompt | self.llm
        adjusted_query = adjustment_chain.invoke({
            "user_query": user_query,
            "sql_query": sql_query,
            "dialect": dialect,
            "table_info": table_info,
            "correction": correction,
            "previous_queries": previous_queries
        }).content.strip()

        
        if adjusted_query.startswith('### Adjusted SQL Query:'):
            adjusted_query = adjusted_query[len('### Adjusted SQL Query:'):].strip()

        return adjusted_query

## Using Self Refection and Iterative Refinement to improve the generated summary. Refinement Goal is to make the summary more concise and better answer the user query.
class ResponseSummarizer:
    def __init__(self, llm):
        self.llm = llm

    def summarize(self, user_query: str, dataframe) -> str:
        summary_agent_template = """You are a data analyst. Given a user query and a pandas DataFrame, summarize the data in a user-readable format.
        Describe the key insights, trends, and any notable observations from the data that answer the user query.
        Make sure to include statistics, comparisons, and any relevant details that provide a clear understanding of the data.

        User Query: {user_query}
        DataFrame:
        {dataframe}
        Summary:
        """
        summary_agent_prompt = PromptTemplate.from_template(summary_agent_template)
        summary_chain = summary_agent_prompt | self.llm
        summary = summary_chain.invoke({"dataframe": dataframe.to_dict(), "user_query": user_query}).content.strip()

        summary = self.iterative_refinement(user_query, summary)

        # print("\n--- Final Summary ---\n", summary)
        return summary

    def iterative_refinement(self, user_query: str, summary: str, max_iterations = 3) -> str:
        for _ in range(max_iterations):
            reflection_result = self.self_reflect(user_query, summary)
            # print("\n--- Summary ---\n", summary)
            # print("\n--- Reflection ---\n", reflection_result)
            if reflection_result.lower().strip() == "all good":
                break
            summary = self.adjust_summary(user_query, summary, reflection_result)
        return summary

    def self_reflect(self, user_query: str, summary: str) -> str:
        reflection_template = """You are a data analyst. Given a user query and a generated summary, evaluate the summary to ensure it is concise and answers the user query in natural language.
        If the summary is already good and no improvements are needed, your response should just be one word: "All good". Otherwise, provide feedback and suggest improvements. Do not include an improved summary.

        User Query: {user_query}
        Generated Summary: {summary}
        Reflection:
        """
        reflection_prompt = PromptTemplate.from_template(reflection_template)
        reflection_chain = reflection_prompt | self.llm
        reflection_result = reflection_chain.invoke({"user_query": user_query, "summary": summary}).content.strip()

        return reflection_result

    def adjust_summary(self, user_query: str, summary: str, reflection: str) -> str:
        adjustment_template = """You are a data analyst. Given a user query, a generated summary, and reflection feedback, adjust the summary to make it more concise and better answer the user query.

        User Query: {user_query}
        Generated Summary: {summary}
        Reflection Feedback: {reflection}
        Adjusted Summary:
        """
        adjustment_prompt = PromptTemplate.from_template(adjustment_template)
        adjustment_chain = adjustment_prompt | self.llm
        adjusted_summary = adjustment_chain.invoke({"user_query": user_query, "summary": summary, "reflection": reflection}).content.strip()


        if adjusted_summary.startswith('### Adjusted Summary:'):
            adjusted_summary = adjusted_summary[len('### Adjusted Summary:'):].strip()

        return adjusted_summary

## Don't see much improvements with self-reflection with this. Might remove later and load from CustomAgents_v5.py
class AnalystAgent:
    def __init__(self, llm):
        self.llm = llm

    def generate_viz_description(self, user_query: str, head: str, info: str, describe: str) -> str:
        analyst_agent_template = """
        You are a data visualization expert.
        Given a description of a dataset and its key characteristics, generate a concise, instructive description for a data visualization that would best represent the data.
        Include any necessary preprocessing and statistical analysis steps that would help in better answering the user query.
        The description should be paragraphs and should include clear instructions on how to create the visualization, including the type of chart, axis labels, and any specific insights or comparisons to highlight.
        Ensure the instructions are for creating just one visualization.
        Do not just use basic visualizations; make impressive visualizations that effectively communicate the data insights.
        Do not write or suggest any code. Just provide clear and detailed instructions for the user to understand the whole process.
        Use the following information to generate the visualization description:

        Dataset Information:
        - User query for the data: {user_query}
        - DataFrame Head (df.head()): {head}
        - DataFrame Info (df.info()): {info}
        - DataFrame Description (df.describe()): {describe}

        Description:
        """
        analyst_agent_prompt = PromptTemplate.from_template(analyst_agent_template)
        analyst_chain = analyst_agent_prompt | self.llm
        viz_desc = analyst_chain.invoke({
            "head": head,
            "info": info,
            "describe": describe,
            "user_query": user_query
        }).content.strip()

        # print("\n--- Generated Visualization Description ---\n", viz_desc)
        viz_desc = self.self_reflect(user_query, viz_desc)

        return viz_desc

    def self_reflect(self, user_query: str, viz_desc: str) -> str:
        reflection_template = """You are a data visualization expert. Given a user query and a generated visualization description, evaluate the visualization description to ensure it effectively answers the user query.
        Ensure the instructions are for creating just one visualization.
        Consider the following:
        - Is there a better method or visualization that answers the query?
        - Can any improvements be made to the current visualization?
        - Aim for impressive visualizations that effectively communicate the data insights, even if the code is complex.
        Do not include an improved description and do not write or suggest any code. Just provide feedback on the existing description and suggest improvements.
        If the visualization description is already good and needs no improvement, your response should be just one word: "All good". Otherwise, provide feedback and suggest improvements.
        
        User Query: {user_query}
        Generated Visualization Description: {viz_desc}
        Reflection:
        """
        reflection_prompt = PromptTemplate.from_template(reflection_template)
        reflection_chain = reflection_prompt | self.llm
        reflection_result = reflection_chain.invoke({
            "user_query": user_query,
            "viz_desc": viz_desc
        }).content.strip()

        # print("\n--- Reflection ---\n", reflection_result)
        if reflection_result.lower().strip() != "all good":
            viz_desc = self.adjust_description(user_query, viz_desc, reflection_result)
            # print("\n--- Adjusted Visualization Description ---\n", viz_desc)

        return viz_desc

    def adjust_description(self, user_query: str, viz_desc: str, reflection: str) -> str:
        adjustment_template = """
        You are a data visualization expert. Given a user query, a generated visualization description, reflection feedback, adjust the visualization description to make it better answer the user query.
        Ensure the instructions are for creating just one visualization.


        User Query: {user_query}
        Generated Visualization Description: {viz_desc}
        Reflection Feedback: {reflection}
        Adjusted Visualization Description:
        """
        adjustment_prompt = PromptTemplate.from_template(adjustment_template)
        adjustment_chain = adjustment_prompt | self.llm
        adjusted_viz_desc = adjustment_chain.invoke({
            "user_query": user_query,
            "viz_desc": viz_desc,
            "reflection": reflection
        }).content.strip()

        if adjusted_viz_desc.startswith('### Adjusted Visualization Description:'):
            adjusted_viz_desc = adjusted_viz_desc[len('### Adjusted Visualization Description:'):].strip()

        return adjusted_viz_desc

class VisualizationAgent:
    def __init__(self, llm):
        self.llm = llm

    def generate_viz_code(self, description: str, dataframe) -> str:
        viz_agent_template = """
        You are a data visualization expert. Given a description of the desired visualization and a pandas DataFrame, generate just the Python code to create the visualization using plotly.
        You can only use the following libraries: numpy (as np), pandas (as pd), plotly.graph_objects (as go).
        Do not include any import statements or definitions of the dataset. The dataset is available as df, and the modules are already imported.
        The generated code should just create the visualization figure object as fig and not return it or show it.
        Ensure the code is complete and correct.
        The visualization should include:
        - Proper legends for different data series.
        - Axis titles for both the x-axis and y-axis.
        - A figure title that describes the visualization.
        - Use pastel colors for the different data series.

        Description: {description}
        DataFrame: {dataframe}
        Visualization Code:
        """
        viz_agent_prompt = PromptTemplate.from_template(viz_agent_template)
        viz_chain = viz_agent_prompt | self.llm
        viz_code = viz_chain.invoke({"description": description, "dataframe": dataframe.to_dict()}).content.strip()
        viz_code = viz_code.replace('`', '').strip()
        if viz_code.startswith('python'): viz_code = viz_code[len('python'):].strip()
        return viz_code


## Tried many times but self correction proved to be fault. Running out of OpenAI balance, Will try again with client's API.

# class VisualizationAgent:
#     def __init__(self, llm):
#         self.llm = llm

#     def generate_viz_code(self, description: str, dataframe) -> str:
#         viz_agent_template = """
#         You are a data visualization expert. Given a description of the desired visualization and a pandas DataFrame, generate just the Python code to create the visualization using plotly.
#         You can only use the following libraries: numpy (as np), pandas (as pd), plotly.graph_objects (as go).
#         Do not include any import statements or definitions of the dataset. The dataset is available as df, and the modules are already imported.
#         The generated code should just create the visualization figure object as fig and not return it or show it.
#         Ensure the code is complete and correct.
#         The visualization should include:
#         - Proper legends for different data series.
#         - Axis titles for both the x-axis and y-axis.
#         - A figure title that describes the visualization.
#         - Use pastel colors for the different data series.

#         Description: {description}
#         DataFrame: {dataframe}
#         Visualization Code:
#         """
#         viz_agent_prompt = PromptTemplate.from_template(viz_agent_template)
#         viz_chain = viz_agent_prompt | self.llm
#         viz_code = viz_chain.invoke({"description": description, "dataframe": dataframe.to_dict()}).content.strip()

#         print("\n--- Generated Visualization Code ---\n", viz_code)
#         viz_code = self.reflect(description, viz_code, dataframe)
#         viz_code = viz_code.replace('`', '').strip()
#         if viz_code.startswith('python'): viz_code = viz_code[len('python'):].strip()

#         return viz_code

#     def reflect(self, description: str, viz_code: str, dataframe) -> str:
#         combined_template = """
#         You are a data visualization expert. Given a description of the desired visualization, the generated Python code, and a pandas DataFrame, evaluate the code to ensure it is complete, correct, and meets the desired criteria.
#         Check for any errors or issues that might cause wrong or ambiguous visualizations. Correct any identified issues and ensure the code is complete and correct.
#         If the code is already good and needs no improvement, your response should be just one word: "All good". Otherwise, provide feedback and suggest improvements.
#         Consider if there is a better method/visualization that answers the query or if any improvements can be made in the current visualization.
#         Do not suggest any code in your response, just provide instructions.

#         Description: {description}
#         Visualization Code: {viz_code}
#         DataFrame: {dataframe}
#         Reflection and Correction:
#         """
#         reflect_prompt = PromptTemplate.from_template(combined_template)
#         reflect_chain = reflect_prompt | self.llm
#         reflect_result = reflect_chain.invoke({
#             "description": description,
#             "viz_code": viz_code,
#             "dataframe": dataframe.to_dict()
#         }).content.strip()

#         if reflect_result.lower().strip() != "all good":
#             viz_code = self.adjust_code(description, viz_code, reflect_result, dataframe)
#             print("\n--- Reflection and Correction ---\n", reflect_result)
#             print("\n--- Adjusted Visualization Code ---\n", viz_code)

#         return viz_code

#     def adjust_code(self, description: str, viz_code: str, reflection: str, dataframe) -> str:
#         adjustment_template = """
#         You are a data visualization expert. Given a description of the desired visualization, the generated Python code, reflection feedback, and a pandas DataFrame, adjust the code to make it better answer the user query.
#         You can only use the following libraries: numpy (as np), pandas (as pd), plotly.graph_objects (as go).
#         Do not include any import statements or definitions of the dataset. The dataset is available as df, and the modules are already imported.
#         The generated code should just create the visualization figure object as fig and not return it or show it.
#         Ensure the code is complete and correct.
#         The visualization should include:
#         - Proper legends for different data series.
#         - Axis titles for both the x-axis and y-axis.
#         - A figure title that describes the visualization.
#         - Use pastel colors for the different data series.        

#         Description: {description}
#         Visualization Code: {viz_code}
#         Reflection Feedback: {reflection}
#         DataFrame: {dataframe}
#         Adjusted Visualization Code:
#         """
#         adjustment_prompt = PromptTemplate.from_template(adjustment_template)
#         adjustment_chain = adjustment_prompt | self.llm
#         adjusted_viz_code = adjustment_chain.invoke({
#             "description": description,
#             "viz_code": viz_code,
#             "reflection": reflection,
#             "dataframe": dataframe.to_dict()
#         }).content.strip()

#         if adjusted_viz_code.startswith('### Adjusted Visualization Code:'):
#             adjusted_viz_code = adjusted_viz_code[len('### Adjusted Visualization Code:'):].strip()

#         return adjusted_viz_code

