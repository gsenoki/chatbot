from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
import getpass
import os
import json
import csv
from sentence_transformers import SentenceTransformer, util

# function that calculates the semantic similarity of the chatbot response to the expect answer
def semantic_similarity(response, expected):
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
    embeddings = model.encode([response, expected], convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1])
    return similarity.item()

# gets the api key of groq
os.environ["GROQ_API_KEY"] = getpass.getpass()

# loads the tests dataets in json
with open("test_dataset_3.json", "r", encoding="utf-8") as f:
    test_dataset = json.load(f)

# read the text file normas unicamp
with open("normas_unicamp_noLines.txt", "r", encoding="utf-8") as file:
    content = file.read()

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

# test all cases in the test dataset
results = []
for test in test_dataset:
    query = test["query"]
    expected = test["expected_answer"]
    response = qa.run(query)
    similarity = semantic_similarity(response, expected)
results.append({"query": query, "response": response, "expected": expected, "similarity": similarity})

# Print results of the test database
for result in results:
    print(f"Query: {result['query']}")
    print(f"Response: {result['response']}")
    print(f"Expected: {result['expected']}")
    print(f"similarity: {result['similarity']}")
    print("---")

# print median accuracy
accuracy = sum([result["similarity"] for result in results]) / len(results)
print(f"Chatbot Accuracy: {accuracy * 100:.2f}%")

# saves the results in an cvs file
with open("test_results_3.csv", "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["query", "response", "expected", "similarity"])
    writer.writeheader()
    writer.writerows(results)