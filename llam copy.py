import streamlit as st 
from streamlit_chat import message
import tempfile
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
from langchain.chains import LLMChain
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.callbacks import get_openai_callback
from langchain.llms import AzureOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain import HuggingFacePipeline
import transformers
from transformers import AutoTokenizer,AutoModelForCausalLM
import torch
from langchain.chains import RetrievalQA
from langchain.llms import LlamaCpp
# from ctransformers import AutoModelForCausalLM
from transformers import pipeline
import pickle
from PIL import Image

DB_FAISS_PATH = 'models/db_faiss'

#Loading the model
@st.cache_resource
def load_llm():
    # Load the locally downloaded model here
    # model = "tiiuae/falcon-7b-instruct"
    # # # # tokenizer = AutoTokenizer.from_pretrained("togethercomputer/LLaMA-2-7B-32K")
    # # # # model = AutoModelForCausalLM.from_pretrained("togethercomputer/LLaMA-2-7B-32K", trust_remote_code=True, torch_dtype=torch.float16)
    # tokenizer = AutoTokenizer.from_pretrained(model)
    # device = torch.device("cpu")
    # pipe = transformers.pipeline(
    #     "text-generation",
    #     model=model,
    #     tokenizer=tokenizer,
    #     torch_dtype=torch.bfloat16,
    #     trust_remote_code=True,
    #     max_length=512,
    #     temperature=0,
    #     top_p=0.95,
    #     top_k=10,         
    #     repetition_penalty=1.15,
    #     num_return_sequences=1,
    #     pad_token_id=tokenizer.eos_token_id
        
    # )
    
    # model_repo = 'daryl149/llama-2-7b-chat-hf'
        
    # tokenizer = AutoTokenizer.from_pretrained(model_repo, use_fast=True)

    # model = AutoModelForCausalLM.from_pretrained("daryl149/llama-2-7b-chat-hf",device_map='auto',
    #     torch_dtype=torch.float16,
    #     low_cpu_mem_usage=True,
    #     trust_remote_code=True
    # )
    
    # max_len = 2048
    # pipe = pipeline(
    # task = "text-generation",
    # model = model,
    # tokenizer = tokenizer,
    # pad_token_id = tokenizer.eos_token_id,
    # max_length = max_len,
    # temperature = 0,
    # top_p = 0.95,
    # repetition_penalty = 1.15
    # )

    # local_llm = HuggingFacePipeline(pipeline=pipe)
    os.environ["OPENAI_API_TYPE"] = "azure"
    os.environ["OPENAI_API_VERSION"] = "2023-03-15-preview"
    os.environ["OPENAI_API_BASE"] = "https://extractopenai.openai.azure.com/"
    os.environ["OPENAI_API_KEY"] = "288db7242d9b4147946a778268f65991"
    llm1 = AzureOpenAI(
        deployment_name="ContractManagement",
        model_name="text-davinci-003",
        temperature=0.1,

    )
    return llm1
llm = load_llm()
img = Image.open("logo.png")
st.image(img,width=200)
img1 =  Image.open("avatar1.png")
template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Context: {context}
Question: {question}
Only return the helpful answer below and nothing else.
Helpful answer:
"""


pdf = st.sidebar.file_uploader("Upload your Data", type="pdf")

if pdf is not None:    
 
    pdf_reader = PdfReader(pdf)          
    
    store_name = pdf.name[:-4]        

    if os.path.exists(f"{store_name}.pkl"):
        with open(f"{store_name}.pkl","rb") as f:
            vectorstore = pickle.load(f)
        # #st.write("Already, Embeddings loaded from the your folder (disks)")
        print("available")
    else:
        print("embeddings")
                
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=800,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        
        # Convert the chunks of text into embeddings to form a knowledge base
        # embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
        knowledgeBase = FAISS.from_texts(chunks, embeddings)
        with open(f"{store_name}.pkl","wb") as f:
            pickle.dump(knowledgeBase,f)    
            
            knowledgeBase.save_local(DB_FAISS_PATH)
        with open(f"{store_name}.pkl","rb") as f:
            vectorstore = pickle.load(f)
    prompt = PromptTemplate(
    template=template,
    input_variables=['context', 'question'])
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(),combine_docs_chain_kwargs={"prompt": prompt},memory=memory)
    

    def conversational_chat(query):
        result = chain({"question": query, "chat_history": st.session_state['history']})
        st.session_state['history'].append((query, result["answer"]))
        
        return result["answer"]
    
    if 'history' not in st.session_state:
        st.session_state['history'] = []


    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello ! Ask me anything about " + pdf.name + " 🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey ! 👋"]
        
    #container for the chat history
    response_container = st.container()
    #container for the user's text input
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            
            user_input = st.text_input("Query:", placeholder="Talk to your csv data here (:", key='input')
            submit_button = st.form_submit_button(label='Send')
            
        if submit_button and user_input:
            output = conversational_chat(user_input)
            
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")



    