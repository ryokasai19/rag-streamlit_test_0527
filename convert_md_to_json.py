import json
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load and split Markdown
loader = TextLoader("sample.md")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# Convert to list of plain strings
chunk_texts = [doc.page_content for doc in chunks]

# Save to JSON
with open("sample_chunks.json", "w", encoding="utf-8") as f:
    json.dump(chunk_texts, f, ensure_ascii=False, indent=2)

print("✅ Saved pre-chunked file to sample_chunks.json")