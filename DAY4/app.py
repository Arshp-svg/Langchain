#Document qna bot using Llama2 ,langchain and GroqAPI
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
import time

# Load env variables
load_dotenv()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")
groq_api_key = os.getenv("GROQ_API_KEY")

# Streamlit UI
st.title("📄 Document QnA Bot with Llama 2 and GroqAPI")
llm=ChatGroq(api_key=groq_api_key, model="llama-3.3-70b-versatile")

promt=ChatPromptTemplate.from_template(
    """answer the questions based on the provided context only.
    please provide the most relevant answer based on the context.
    <context>
    {context}
    </context>
    Question: {input}"""  # Changed from {question} to {input}
                                       
)

prompt1=st.text_input("Enter your question from documents")

def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings= HuggingFaceEmbeddings()
        st.session_state.loader= PyPDFDirectoryLoader("DAY3/us_sensus", glob="*.pdf") ##data ingestion
        st.session_state.documents= st.session_state.loader.load() #loading documents
        st.session_state.text_splitter= RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200) #splitting documents into chunks
        st.session_state.chunks= st.session_state.text_splitter.split_documents(st.session_state.documents[:50]) #splitting documents
        st.session_state.vectors= FAISS.from_documents(st.session_state.chunks, st.session_state.embeddings) #creating vector store
    
if st.button("Get Answer"):
    vector_embedding()
    st.write("fetching answer from documents...")
    
if prompt1:
    # Check if vectors exist before proceeding
    if "vectors" not in st.session_state:
        vector_embedding()
    
    chain=create_stuff_documents_chain(llm=llm, prompt=promt)
    retriver=st.session_state.vectors.as_retriever()
    doc_chain=create_retrieval_chain(retriever=retriver, combine_docs_chain=chain)  # Fixed typo: combine_docs_chain instead of docs_chain
    start=time.process_time()
    response=doc_chain.invoke({"input": prompt1}) 
    print("Response time:", time.process_time()-start)
    st.write(response['answer'])
    
    with st.expander("Source Documents"):
        for i, source in enumerate(response['context']):
            st.write(source.page_content)
            st.write("---")