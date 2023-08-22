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
from langchain import HuggingFacePipeline
import transformers
from transformers import AutoTokenizer,AutoModelForCausalLM
import torch
from langchain.chains import RetrievalQA
from langchain.llms import LlamaCpp
from ctransformers import AutoModelForCausalLM

import pickle
from PIL import Image

# os.environ["OPENAI_API_TYPE"] = "azure"
# os.environ["OPENAI_API_VERSION"] = "2023-03-15-preview"
# os.environ["OPENAI_API_BASE"] = "https://extractopenai.openai.azure.com/"
# os.environ["OPENAI_API_KEY"] = "288db7242d9b4147946a778268f65991"
# llm = AzureOpenAI(
#     deployment_name="chatdataset",
#     model_name="gpt-35-turbo",
#     temperature=0.1,
    
# )
# Load environment variables
@st.cache_resource
def load():
    # os.environ["OPENAI_API_TYPE"] = "azure"
    # os.environ["OPENAI_API_VERSION"] = "2023-03-15-preview"
    # os.environ["OPENAI_API_BASE"] = "https://extractopenai.openai.azure.com/"
    # os.environ["OPENAI_API_KEY"] = "288db7242d9b4147946a778268f65991"
    # llm1 = AzureOpenAI(
    #     deployment_name="ContractManagement",
    #     model_name="text-davinci-003",
    #     temperature=0.1,

    # )

    # llm1 = LlamaCpp(model_path='models/LLaMA-2-7B-32K.ggmlv3.q8_0.bin',
    #                         temperature=0.8,
    #                         n_threads=8,
    #                         n_ctx=32000,
    #                         n_batch=512,
    #                         max_tokens=10000)
    # # This is for illustration purposes only
    model = "tiiuae/falcon-7b-instruct"
    # tokenizer = AutoTokenizer.from_pretrained("togethercomputer/LLaMA-2-7B-32K")
    # model = AutoModelForCausalLM.from_pretrained("togethercomputer/LLaMA-2-7B-32K", trust_remote_code=True, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model)
    pipe = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        max_length=2048,
        temperature=0,
        top_p=0.95,
        top_k=10,         
        repetition_penalty=1.15,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id
        
    )
#     LLM = AutoModelForCausalLM.from_pretrained(
#     'D:/llama32/models/LLaMA-2-7B-32K.ggmlv3.q8_0.bin',
#     model_type="llama",
# )

    local_llm = HuggingFacePipeline(pipeline=pipe)
#     ll = HuggingFacePipeline(pipeline = pipe)
    # pdf_reader = PdfReader("c00757358.pdf")
    # text = ""
    # for page in pdf_reader.pages:
    #     text += page.extract_text()
    # text_splitter = CharacterTextSplitter(
    #     separator="\n",
    #     chunk_size=1000,
    #     chunk_overlap=200,
    #     length_function=len
    # )
    # chunks = text_splitter.split_text(text)
    
    # Convert the chunks of text into embeddings to form a knowledge base
    # embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # knowledgeBase = FAISS.from_texts(chunks, embeddings)
    # with open(f"pdfdata.pkl","wb") as f:
    #     pickle.dump(knowledgeBase,f)    
    
    
    return local_llm

# llm1 = load1()

llm = load()
load_dotenv()

def process_text(text):
    # Split the text into chunks using langchain
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    # Convert the chunks of text into embeddings to form a knowledge base
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
    knowledgeBase = FAISS.from_texts(chunks, embeddings)




def main():
    img = Image.open("logo.png")
    st.image(img,width=200)
 
    
    pdf = st.file_uploader('Upload your PDF Document', type='pdf')
    
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
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_text(text)
            
            # Convert the chunks of text into embeddings to form a knowledge base
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            knowledgeBase = FAISS.from_texts(chunks, embeddings)
            with open(f"{store_name}.pkl","wb") as f:
                pickle.dump(knowledgeBase,f)    
            with open(f"{store_name}.pkl","rb") as f:
                vectorstore = pickle.load(f)
        
            
        

        query = st.text_input('Ask a question to the PDF')     

              
        
        if st.button("Answer Me") and query:
            template = """You are a chatbot having a conversation with a human.

            Given the following extracted parts of a long document and a question, create a final answer.

            {context}

            
            Human: {human_input}
            Chatbot:"""
            # with open(f"pdfdata.pkl","rb") as f:
            #     vectorstore = pickle.load(f)
            with open(f"{store_name}.pkl","rb") as f:
                vectorstore = pickle.load(f)
       
            
            docs = vectorstore.similarity_search(query)
            # print(docs)
            # prompt1 = PromptTemplate(input_variables=["human_input", "context"], template=template)
            # llm_chain=  LLMChain(llm=llm,prompt=prompt1)
            chain = load_qa_chain(llm, chain_type='stuff')
            # chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docs.as_retriever())
            print(docs)
            response = chain({"input_documents": docs, "question": query})
            # response = llm_chain.run({"context": docs,"human_input": query})
               
            st.write(response["output_text"])
            
            
if __name__ == "__main__":
    main()
