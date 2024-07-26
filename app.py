# Importing Required Libraries
import time
import streamlit as st
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_objectbox.vectorstores import ObjectBox
from langchain_community.document_loaders import PyPDFDirectoryLoader

import os
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Set API keys from environment variables
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

# Streamlit app configuration
st.set_page_config(
    page_title="Pakistan Budget 2024-2025 Document Query",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="auto",
)

# Custom CSS for buttons and layout
st.markdown("""
    <style>
    .stButton button {
        background-color: #4CAF50; /* Green */
        color: white;
        border-radius: 12px;
        padding: 10px 24px;
        margin: 10px auto;
        display: block;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    .centered {
        display: flex;
        justify-content: center;
    }
    </style>
""", unsafe_allow_html=True)

# Streamlit app title
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>📊 Pakistan Budget 2024-2025 Document Query 📊</h1>", unsafe_allow_html=True)

# Initialize the language model
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Questions: {input}
    """
)

# Function to create vector embeddings and store them in ObjectBox
def vector_embeddings():
    with st.spinner('Embedding documents...'):
        if "vectors" not in st.session_state:
            st.session_state.embeddings = OpenAIEmbeddings()
            st.session_state.loader = PyPDFDirectoryLoader("./budget") # Data Ingestion
            st.session_state.docs = st.session_state.loader.load() # Loading documents from folder
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[38:59]) # Embedding document from page 38 to 59
            st.session_state.vectors = ObjectBox.from_documents(
                st.session_state.final_documents,
                st.session_state.embeddings,
                embedding_dimensions=770
            )
    st.success("📄 Document embeddings are ready. You can now ask questions!")

# User interface for embedding documents and asking questions
if st.button("Initialize Document Embeddings"):
    vector_embeddings()

input_prompt = st.text_input("🔍 Enter your question about the budget...")

if st.button("Submit Query"):
    if input_prompt:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        start = time.process_time()
        response = retrieval_chain.invoke({"input": input_prompt})
        processing_time = time.process_time() - start
        
        st.markdown(f"<h2 style='color: #4CAF50;'>🔎 Response:</h2>", unsafe_allow_html=True)
        st.write(f"{response['answer']}")
        
        with st.expander("📚 Documents Similarity Search"):
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("----------------------------")
