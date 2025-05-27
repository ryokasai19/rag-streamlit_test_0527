import streamlit as st
from dotenv import load_dotenv
import os
import json
from langchain_core.documents import Document

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains.retrieval_qa.base import RetrievalQA


# Load API key from .env or Streamlit Cloud secrets
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))
api_key = os.getenv("OPEN_API_KEY")
print("✅ DEBUG: Loaded API key:", api_key)

# Safety check
if not api_key:
    st.error("❌ OPENAI_API_KEY not found. Please add it to .env or Streamlit Secrets.")
    st.stop()

st.set_page_config(page_title="Ask Your Markdown", page_icon="📘")
st.title("📘 Ask anything about Wanderlust!")



with open("sample_chunks.json", "r", encoding="utf-8") as f:
    chunk_texts = json.load(f)
chunks = [Document(page_content=text) for text in chunk_texts]

# Embed and store in vector DB
embeddings = OpenAIEmbeddings(openai_api_key=api_key)
vectorstore = FAISS.from_documents(chunks, embeddings)

# Setup retrieval chain
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(openai_api_key=api_key),
    retriever=retriever
)

# Ask a question
query = st.text_input("🔍 Ask a question about the document:")

if query:
    result = qa.invoke(query)
    st.subheader("📎 Answer")
    answer = result["result"] 
    st.write(answer)
