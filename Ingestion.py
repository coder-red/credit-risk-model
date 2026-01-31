# ingestion.py - Process EBA PDF and upload to Pinecone
import os
import time
from dotenv import load_dotenv

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index_name = os.environ.get("PINECONE_INDEX_NAME", "eba-credit-risk")

# Check/create index
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

index = pc.Index(index_name)

# Initialize embeddings + vector store
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# Load PDF (put EBA PDF in data/raw/)
loader = PyPDFDirectoryLoader("data/raw/EBA_Methodological_Guide.pdf")
raw_documents = loader.load()

# Split documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)
documents = text_splitter.split_documents(raw_documents)

# Generate IDs and upload
uuids = [f"doc_{i}" for i in range(len(documents))]
vector_store.add_documents(documents=documents, ids=uuids)

print(f"✅ Added {len(documents)} chunks to Pinecone")
print(f"Index stats: {index.describe_index_stats()}")