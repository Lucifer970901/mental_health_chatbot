# import basics
import os
from dotenv import load_dotenv

# import pinecone
from pinecone import Pinecone

# import langchain
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import OllamaEmbeddings

# Load environment variables
load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# Initialize index
index_name = "sample-index"
index = pc.Index(index_name)

# Initialize embeddings model + vector store
embeddings = OllamaEmbeddings(model="llama3.2:1b")
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# Example queries
queries = [
    "What did someone have for breakfast?",
    "Tell me about the weather",
    "Any news about crime?",
    "What's happening with technology?",
    "Any financial news?"
]

# Perform similarity search for each query
print("\nQuerying the vector store...")
print("-" * 50)

for query in queries:
    print(f"\nQuery: {query}")
    results = vector_store.similarity_search(query, k=2)  # Get top 2 most similar documents
    
    print("\nRelevant documents:")
    for doc in results:
        print(f"\nContent: {doc.page_content}")
        print(f"Source: {doc.metadata['source']}")
    print("-" * 50)

#'''

