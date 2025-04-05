import streamlit as st
import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index = pc.Index(os.environ.get("PINECONE_INDEX_NAME"))

# Initialize embeddings and vector store
embeddings = OllamaEmbeddings(model="phi4-mini")
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# Initialize Ollama LLM
llm = OllamaLLM(model="phi4-mini", temperature=0.3)

# Define prompt template
prompt_template = """You are Haven, a supportive AI companion. Follow these guidelines:
1. Respond naturally to greetings (e.g., "Hello! How can I help?")
2. For emotional sharing, ask one open-ended question
3. Keep responses under 50 words
4. Never give medical advice
5. Use this context if relevant: {context}

Chat history: {chat_history}
Human: {question}
Assistant:"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "chat_history", "question"]
)

# Initialize retrieval chain with prompt
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vector_store.as_retriever(),
    return_source_documents=False,
    combine_docs_chain_kwargs={"prompt": PROMPT}
)

# Streamlit interface
st.title("Haven - Your Supportive AI Companion")
st.write("Ask questions, or just talk - I'm here to listen!")

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
        # Print chat history to terminal
        print("\n=== CURRENT CHAT HISTORY ===")
        for i, (user_msg, ai_msg) in enumerate(chat_history, 1):
            print(f"Exchange {i}:")
            print(f"User: {user_msg}")
            print(f"AI: {ai_msg}\n")

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
        #if response["source_documents"]:
        #    with st.expander("Source Documents"):
        #        for doc in response["source_documents"]:
        #            st.markdown(f"**Content:** {doc.page_content[:200]}...")
        #            st.markdown("---")
                    
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.error("Please make sure Ollama is running and the model is available.")