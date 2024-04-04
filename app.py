from flask import Flask, render_template, request
from pymongo import MongoClient
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import os

app = Flask(__name__)

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = "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"

# Define the connection string
MONGO_URL = 'mongodb+srv://tamizhselvanrd:D28xoZ87qXdFcTTc@cluster0.q68kowk.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0'

# Connect to MongoDB
client = MongoClient(MONGO_URL)

# Access the database
db = client["Dream_Nest"]

# Access all collections
collections = db.list_collection_names()

# Download embeddings from OpenAI
embeddings = OpenAIEmbeddings()

# Load question answering model
chain = load_qa_chain(OpenAI(), chain_type="stuff")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']

    # Fetch data from MongoDB Atlas for all collections
    all_documents = []
    for collection_name in collections:
        collection = db[collection_name]
        for doc in collection.find():
            all_documents.append(doc)

    # Prepare text for analysis
    raw_text = "\n".join([doc.get("description", "") for doc in all_documents])

    # Split text into manageable chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=800,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_text(raw_text)

    # Create document search index
    document_search = FAISS.from_texts(texts, embeddings)

    # Search for relevant documents
    docs = document_search.similarity_search(query)

    # Run question answering on the retrieved documents
    answers = chain.run(input_documents=docs, question=query)

    # Convert answers to a list of strings
    answers_list = [str(answer) for answer in answers]

    return render_template('result.html', query=query, answers=answers_list)

if __name__ == '__main__':
    app.run(debug=True)
