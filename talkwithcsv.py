from langchain import OpenAI
from langchain.agents import create_pandas_dataframe_agent
import pandas as pd
from dotenv import load_dotenv 
import json
import streamlit as st
import os
from langchain.llms import AzureOpenAI
import re
import matplotlib.pyplot as plt
from langchain.agents import create_csv_agent
from langchain.llms import GPT4All
import pandas as pd
from langchain.agents import create_pandas_dataframe_agent
from langchain.llms import LlamaCpp
from langchain.callbacks.base import BaseCallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from PIL import Image

callback_manage = BaseCallbackManager([StreamingStdOutCallbackHandler()])

from langchain.llms import AzureOpenAI
load_dotenv()

def csv_tool():

    df = pd.read_csv("D:/chatd/FinalElectricconsumption.csv")
    # os.environ["OPENAI_API_TYPE"] = "azure"
    # os.environ["OPENAI_API_VERSION"] = "2023-03-15-preview"
    # os.environ["OPENAI_API_BASE"] = "https://extractopenai.openai.azure.com/"
    # os.environ["OPENAI_API_KEY"] = "288db7242d9b4147946a778268f65991"
    # llm = AzureOpenAI(
    #  deployment_name="chatdataset",
    #  model_name="gpt-35-turbo",
    #  temperature=0,
    # )
    llm = LlamaCpp(model_path="./models/ggml-vicuna-13b-1.1-q4_2.bin",callback_manager=callback_manage,verbose=True,n_ctx= 2048)
    return create_pandas_dataframe_agent(llm, df, verbose=True)

def ask_agent(agent, query):
    """
    Query an agent and return the response as a string.

    Args:
        agent: The agent to query.
        query: The query to ask the agent.

    Returns:
        The response from the agent as a string.
    """
    # Prepare the prompt with query guidelines and formatting
    prompt = (
        """
        Let's decode the way to respond to the queries. The responses depend on the type of information requested in the query. 

        1. If the query requires a table, format your answer like this:
           {"table": {"columns": ["column1", "column2", ...], "data": [[value1, value2, ...], [value1, value2, ...], ...]}}

        2. For a bar chart, respond like this:
           {"bar": {"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}
               

        Note: We only accommodate one type of charts :  "bar".

        4. For a plain question that doesn't need a chart or table, your response should be:
           {"answer": "Your answer goes here"}

        For example:
           {"answer": "The Product with the highest Orders is '15143Exfo'"}

        5. If the answer is not known or available, respond with:
           {"answer": "I do not know."}

        Return all output as a string. Remember to encase all strings in the "columns" list and data list in double quotes. 
        For example: {"columns": ["Brand", "Consumption"], "data": [["Starbucks", 2315], ["H&M", 2650]]}

        Now, let's tackle the query step by step. Here's the query for you to work on: 
        """
        + query
    )

    # Run the prompt through the agent and capture the response.
    response = agent.run(prompt)

    # Return the response converted to a string.
    return str(response)

def decode_response(response: str) -> dict:
    """This function converts the string response from the model to a dictionary object.

    Args:
        response (str): response from the model

    Returns:
        dict: dictionary with response data
    """
    efooter = response
    if len(re.findall(r"(?<=}(?!.*\})).*", efooter,flags=re.M)) > 0:
        resul = re.search(r"(?<=}(?!.*\})).*",efooter)[0]
    else:
        resul = ""
        
        
    print("dai prasana")
    pres = response.replace(resul,"").replace("'","\"")
    print(pres)
    return json.loads(pres)

def write_answer(response_dict: dict):
    """
    Write a response from an agent to a Streamlit app.

    Args:
        response_dict: The response from the agent.

    Returns:
        None.
    """

    # Check if the response is an answer.
    if "answer" in response_dict:
        print("prasana")
        st.write(response_dict["answer"])

    # Check if the response is a bar chart.
    # Check if the response is a bar chart.
    if "bar" in response_dict:
        data = response_dict["bar"]
        try:
            df_data = {
                    col: [x[i] if isinstance(x, list) else x for x in data['data']]
                    for i, col in enumerate(data['columns'])
                }       
            df = pd.DataFrame(df_data)
            df.set_index(df.columns[0], inplace=True)
            st.bar_chart(df)
        except ValueError:
            print(f"Couldn't create DataFrame from data: {data}")

    if "pie" in response_dict:
        data = response_dict["pie"]
        try:
            df_data = {col: [x[i] for x in data['data']] for i, col in enumerate(data['columns'])}
            df = pd.DataFrame(df_data)
            df.set_index(df.columns[0], inplace=True)
            
            # Calculate values for the pie chart
            values = df.iloc[:, 0].value_counts()
            print(values)
            labels = values.index.tolist()
            
            # Create the pie chart
            fig, ax = plt.subplots()
            ax.pie(values, labels=labels, autopct='%1.1f%%')
            ax.set_aspect('equal')  # Ensure pie chart is circular
            
            # Display the pie chart in Streamlit
            st.pyplot(fig)
            
        except ValueError:
            st.error(f"Couldn't create DataFrame from data: {data}")

# Check if the response is a line chart.
    if "line" in response_dict:
        data = response_dict["line"]
        try:
            df_data = {col: [x[i] for x in data['data']] for i, col in enumerate(data['columns'])}
            df = pd.DataFrame(df_data)
            df.set_index(df.columns[0], inplace=True)
            st.line_chart(df)
        except ValueError:
            print(f"Couldn't create DataFrame from data: {data}")


    # Check if the response is a table.
    if "table" in response_dict:
        data = response_dict["table"]
        df = pd.DataFrame(data["data"], columns=data["columns"])
        st.table(df)
# st.set_page_config(page_title="Sutherland")
st.title("")
img = Image.open("logo.png")
st.image(img,width=200)

# st.write("Please upload your CSV file below.")

# data = st.file_uploader("Upload a CSV" , type="csv")

query = st.text_area("Send a Message")


if st.button("Submit Query", type="primary"):
    # Create an agent from the CSV file.
    agent = csv_tool()

    # Query the agent.
    response = ask_agent(agent=agent, query=query)

    # Decode the response.
    decoded_response = decode_response(response)

    # Write the response to the Streamlit app.
    write_answer(decoded_response)
