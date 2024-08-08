# Data Analyst Assistant

Data Analyst Assistant is a Streamlit-based application that facilitates SQL query generation, data fetching, summary generation, and data visualization. The app utilizes OpenAI's language models and SQL database tools to interactively assist users with their data analysis needs.

## Features

- **SQL Query Generation**: Automatically generates syntactically correct SQL queries based on user input questions.
- **Data Fetching**: Executes the generated SQL queries and fetches data from a PostgreSQL database.
- **Data Summarization**: Provides concise summaries of the fetched data.
- **Data Visualization**: Generates visualizations of the data using Plotly.
- **Interactive Interface**: Allows users to input custom questions or select from sample queries, and toggle various display options.

## Installation

### Prerequisites

- Python 3.8 or higher
- Streamlit
- PostgreSQL
- Required Python libraries: `dotenv`, `numpy`, `pandas`, `plotly`, `ast`, `langchain_community`, `langchain_core`, `langchain_openai`

### Setup

1. **Clone the repository:**

    ```bash
    git clone https://github.com/rajnaruka06/Multi-Agent-Data-Analysis-App.git
    cd Multi-Agent-Data-Analysis-App
    ```

2. **Create a virtual environment and activate it:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required libraries:**

    ```bash
    pip install streamlit python-dotenv numpy pandas plotly langchain_community langchain_core langchain_openai
    ```

4. **Set up environment variables:**

    Create a `.env` file in the project root directory and add the following environment variables:

    ```env
    POSTGRES_USER=your_postgres_username
    POSTGRES_PASSWORD=your_postgres_password
    POSTGRES_HOST=your_postgres_host
    POSTGRES_PORT=your_postgres_port
    ```

5. **Run the Streamlit app:**

    ```bash
    streamlit run app.py
    ```

## Usage

1. **Open the Streamlit app**: Once the app is running, it will open in your default web browser.

2. **Select a sample question or enter your own**: You can choose from pre-defined sample queries or type your custom question in the provided text area.

3. **Toggle options**: Use the sidebar to toggle various display options, such as generating visualizations, summaries, showing SQL queries, and more.

4. **Fetch results**: Click the "Get results" button to fetch and display the results based on the input question.

## Code Overview

### Main Components

- **SQL Query Generation**: Uses the `dba_agent_template` to generate SQL queries.
- **Data Fetching**: Executes SQL queries using the `QuerySQLDataBaseTool`.
- **Data Summarization**: Summarizes the fetched data using the `summary_agent_template`.
- **Data Visualization**: Generates visualization code using the `viz_agent_template` and visual descriptions using the `analyst_agent_template`.
- **Streamlit Interface**: Provides an interactive UI for user input and displaying results.

### Template Descriptions

- `dba_agent_template`: Generates SQL queries from user questions.
- `summary_agent_template`: Summarizes the data in a user-readable format.
- `viz_agent_template`: Generates Python code for data visualizations using Plotly.
- `analyst_agent_template`: Generates descriptions for visualizations based on dataset characteristics.

## Dataset

The app uses the Chinook dataset for demonstration purposes but will work with any dataset. 

## Acknowledgments

- [Streamlit](https://streamlit.io/)
- [OpenAI](https://openai.com/)
- [LangChain](https://langchain.com/)