import streamlit as st
from dotenv import load_dotenv
import os

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.retrieval_qa.base import RetrievalQA


from dotenv import load_dotenv # this gets my OPEN ai API code
load_dotenv() 
print("LOADED KEY:", os.getenv("OPENAI_API_KEY"))


st.set_page_config(page_title="Ask Your Markdown", page_icon="📘")
st.title("📘 Ask Your Markdown")

# Upload markdown file
uploaded_file = st.file_uploader("Upload a Markdown (.md) file", type="md")

if uploaded_file:
    # Save uploaded file temporarily
    with open("temp.md", "wb") as f:
        f.write(uploaded_file.read())

    # Load and split
    loader = TextLoader("temp.md")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    # Embed and store in FAISS
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Build retriever and QA chain
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(), retriever=retriever)

    # Question input
    query = st.text_input("🔍 Ask a question about the document:")

    if query:
        answer = qa.run(query)
        st.subheader("📎 Answer")
        st.write(answer)