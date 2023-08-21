from langchain.agents import create_csv_agent
from langchain.llms import GPT4All
import pandas as pd
from langchain.agents import create_pandas_dataframe_agent
from langchain.llms import LlamaCpp
from langchain.callbacks.base import BaseCallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

callback_manage = BaseCallbackManager([StreamingStdOutCallbackHandler()])


# Imports
import os
import streamlit as st

# from dotenv import load_dotenv
from langchain.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI


from PIL import Image





def main():
    # Define OpenAI API KEY
    from PIL import Image

    # llm = GPT4All(model="./models/ggml-gpt4all-j-v1.3-groovy.bin", n_ctx=512, n_threads=8)

#     llm = LlamaCpp(
#     model_path="./models/ggml-vic7b-q4_0.bin",
#     n_ctx= 2048,
#     verbose=True,
#     use_mlock=True,
#     n_gpu_layers=12,
#     n_threads=4,
#     n_batch=1000
# )
    llm = LlamaCpp(model_path="./models/ggml-vicuna-13b-1.1-q4_2.bin",callback_manager=callback_manage,verbose=True,n_ctx= 2048)
    # Title and description
    # margins_css = """
    # <style>
    #     .main > div {
    #         padding-left: 0rem;
    #         padding-right: 0rem;
    #     }
    # </style>
    # """
    padding_top = 0
    padding_bottom = 10
    padding_left = 1
    padding_right = 10
    # max_width_str = f'max-width: 100%;'
    st.markdown("""
        <style>
               .block-container {
                    padding-top: 1rem;
                    padding-bottom: 0rem;
                    padding-left: 0rem;
                    padding-right: 0rem;
                }
        </style>
        """, unsafe_allow_html=True)
    # st.sidebar.title("Converse with the Database")
    # st.markdown(margins_css, unsafe_allow_html=True)
    st.title("")
    # st.set_page_config(page_title="Sutherland")
    img = Image.open("logo.png")
    st.image(img,width=200)
    # st.write("Upload a CSV file and query answers from your data.")

    # # Upload File
    # file =  st.file_uploader("Upload CSV file",type=["csv"])
    # if not file: st.stop()

    # Read Data as Pandas
    data = pd.read_csv("C:/Users/sprasa3/Desktop/FinalElectricconsumption.csv")

    # Display Data Head
    # st.write("Data Preview:")
    # st.dataframe(data.head()) 

    # Define pandas df agent - 0 ~ no creativity vs 1 ~ very creative
    agent = create_pandas_dataframe_agent(llm,data,verbose=True) 
    
    # Define Generated and Past Chat Arrays
    if 'generated' not in st.session_state: 
        st.session_state['generated'] = []

    if 'past' not in st.session_state: 
        st.session_state['past'] = []

    # CSS for chat bubbles
    chat_bubble_style = \
    """
        .user-bubble {
            background-color: dodgerblue;
            color: white;
            padding: 8px 12px;
            border-radius: 15px;
            display: inline-block;
            max-width: 70%;
        }
        
        .gpt-bubble {
            background-color: #F3F3F3;
            color: #404040;
            padding: 8px 12px;
            border-radius: 15px;
            display: inline-block;
            max-width: 70%;
            text-align: right;
        }
    """

    # Apply CSS style
    st.write(f'<style>{chat_bubble_style}</style>', unsafe_allow_html=True)

    # Accept input from user
    query = st.text_input("Enter a query:") 

    # Execute Button Logic
    if st.button("Execute") and query:
        with st.spinner('Generating response...'):
            try:
                answer = agent.run(query)

                # Store conversation
                st.session_state.past.append(query)
                st.session_state.generated.append(answer)

                # Display conversation in reverse order
                for i in range(len(st.session_state.past)-1, -1, -1):
                    st.write(f'<div class="gpt-bubble">{st.session_state.generated[i]}</div>', unsafe_allow_html=True)
                    st.write(f'<div class="user-bubble">{st.session_state.past[i]}</div>', unsafe_allow_html=True)
                    st.write("")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

  
if __name__ == "__main__":
    main()   





