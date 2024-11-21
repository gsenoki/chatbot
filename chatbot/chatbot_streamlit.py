import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
import requests
import getpass
import os
import json
import csv
from sentence_transformers import SentenceTransformer, util

st.title("🦜🔗 Chatbot")

groq_api_key = st.sidebar.text_input("GROQ API Key", type="password")

os.environ["GROQ_API_KEY"] = groq_api_key

url = 'https://drive.google.com/file/d/1LxIK5YBypuW3qKyrIYae2oXg6XV4Jd4O/view?usp=sharing'
response = requests.get(url)
content = response.text

text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
docs = text_splitter.create_documents([content])

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
db = FAISS.from_documents(docs, embeddings)

llm = ChatGroq(model="llama3-8b-8192", temperature=0.7)
retriever = db.as_retriever()
qa = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=retriever)

with st.form("my_form"):
    text = st.text_area(
        "Faça sua pergunta sobre o Vestibular Unicamp 2025:",
    )
    submitted = st.form_submit_button("Submit")
    response = qa.run(text)
    st.info(response)

