from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
import requests
import getpass
import os
import json
import csv
from sentence_transformers import SentenceTransformer, util


# gets the api key of groq
os.environ["GROQ_API_KEY"] = getpass.getpass()

# read the text file normas unicamp
url = 'https://drive.google.com/file/d/1LxIK5YBypuW3qKyrIYae2oXg6XV4Jd4O/view?usp=sharing'
response = requests.get(url)
content = response.text
#with open("normas_unicamp_noLines.txt", "r", encoding="utf-8") as file:
#    content = file.read()

# slits the text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
docs = text_splitter.create_documents([content])

# vector database and sentence embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
db = FAISS.from_documents(docs, embeddings)

# cconfigure the chatbot model
llm = ChatGroq(model="llama3-8b-8192", temperature=0.7)
retriever = db.as_retriever()
qa = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=retriever)

# interact with the chatbot
while True:
    query = input("Faça sua pergunta sobre o Vestibular Unicamp 2025: ")
    response = qa.run(query)
    print("Resposta:", response)
