# import basics
import os
import time
from dotenv import load_dotenv

# import pinecone
from pinecone import Pinecone, ServerlessSpec

# import langchain
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.documents import Document

#documents
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

print("Loading environment variables...")
load_dotenv() 

print("Initializing Pinecone...")
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# initialize pinecone database
index_name = os.environ.get("PINECONE_INDEX_NAME")
print(f"Using index name: {index_name}")

# check whether index exists, and create if not
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
print(f"Existing indexes: {existing_indexes}")

if index_name not in existing_indexes:
    print(f"Creating new index '{index_name}'...")
    pc.create_index(
        name=index_name,
        dimension=2048,  # dimension for llama3.2:1b
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(index_name).status["ready"]:
        print("Waiting for index to be ready...")
        time.sleep(1)
    print("Index created successfully!")
else:
    print(f"Index '{index_name}' already exists")

print("Connecting to index...")
index = pc.Index(index_name)

# initialize embeddings model + vector store
print("Initializing Ollama embeddings...")
embeddings = OllamaEmbeddings(model="llama3.2:1b")

print("Creating vector store...")
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
documents_dir = os.path.join(script_dir, "documents")

print(f"Loading PDF documents from {documents_dir}...")
print(f"Checking if directory exists: {os.path.exists(documents_dir)}")
print(f"Directory contents: {os.listdir(documents_dir)}")

loader = PyPDFDirectoryLoader(documents_dir)

try:
    raw_documents = loader.load()
    print(f"Loaded {len(raw_documents)} documents")
    if len(raw_documents) > 0:
        print("First document preview:")
        print(raw_documents[0].page_content[:200] + "...")
except Exception as e:
    print(f"Error loading documents: {str(e)}")
    exit(1)

# splitting the documents
print("Splitting documents into chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)

documents = text_splitter.split_documents(raw_documents)
print(f"Created {len(documents)} document chunks")
if len(documents) > 0:
    print("First chunk preview:")
    print(documents[0].page_content[:200] + "...")

# generate unique id's
print("Generating unique IDs...")
i = 0
uuids = []

while i < len(documents):
    i += 1
    uuids.append(f"id{i}")

# add to database
print("Adding documents to vector store...")
try:
    vector_store.add_documents(documents=documents, ids=uuids)
    print("Documents added successfully!")
except Exception as e:
    print(f"Error adding documents to vector store: {str(e)}")
    exit(1)