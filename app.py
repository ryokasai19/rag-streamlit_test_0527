import streamlit as st
from dotenv import load_dotenv
import os

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains.retrieval_qa.base import RetrievalQA

# Load API key from .env or Streamlit Cloud secrets
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Safety check
if not api_key:
    st.error("❌ OPENAI_API_KEY not found. Please add it to .env or Streamlit Secrets.")
    st.stop()

st.set_page_config(page_title="Ask Your Markdown", page_icon="📘")
st.title("📘 Ask Your Markdown")
st.write("Upload a Markdown file and ask questions about its contents using GPT.")

# File uploader
uploaded_file = st.file_uploader("📄 Upload a Markdown (.md) file", type="md")

if uploaded_file:
    # Save uploaded file temporarily
    with open("temp.md", "wb") as f:
        f.write(uploaded_file.read())

    # Load and chunk document
    loader = TextLoader("temp.md")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

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
        answer = qa.invoke(query)
        st.subheader("📎 Answer")
        st.write(answer)
