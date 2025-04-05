#import streamlit
import streamlit as st
import os
from dotenv import load_dotenv

# import pinecone
from pinecone import Pinecone

# import langchain
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import ConversationalRetrievalChain

# Load environment variables
load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index = pc.Index(os.environ.get("PINECONE_INDEX_NAME"))

# Initialize embeddings and vector store
embeddings = OllamaEmbeddings(model="phi4-mini")
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# Initialize Ollama LLM
llm = OllamaLLM(model="phi4-mini")

# Initialize retrieval chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vector_store.as_retriever(),
    return_source_documents=True,
)

# Streamlit interface
st.title("RAG Chatbot")
st.write("Ask questions about the research paper!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What's your question?"):
    # Display user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    try:
        # Get response from chatbot
        chat_history = [(m["content"], r["content"]) 
                       for m, r in zip(st.session_state.messages[::2], st.session_state.messages[1::2])]
        
        response = qa_chain({
            "question": prompt,
            "chat_history": chat_history
        })
        answer = response["answer"]
        
        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

        # Display source documents
        if response["source_documents"]:
            with st.expander("Source Documents"):
                for doc in response["source_documents"]:
                    st.markdown(f"**Content:** {doc.page_content[:200]}...")
                    st.markdown("---")
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.error("Please make sure Ollama is running and the model is available.")

