import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_ollama import OllamaLLM
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()
os.environ['GROQ_API_KEY'] =os.getenv("GROQ_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")


if "vector" not in st.session_state:
    st.session_state.embeddings = OllamaEmbeddings(model="gemma:2b")
    st.session_state.loader = WebBaseLoader("https://python.langchain.com/docs/introduction/")
    st.session_state.docs= st.session_state.loader.load()
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
    st.session_state.vector = FAISS.from_documents(st.session_state.documents, st.session_state.embeddings)
    
    
    
st.title("📚 Document Q&A with LangChain and Ollama")
llm=ChatGroq(
    groq_api_key=groq_api_key,
    model="gemma2-9b-it", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "answer the question based on the context provided only."),
    ("user", "<context>\ncontext: {context}\n</context>\nquestion: {input}")
])

document_chain= create_stuff_documents_chain(llm, prompt=prompt)
retriever=st.session_state.vector.as_retriever()
retrieval_chain=create_retrieval_chain(retriever,document_chain)

prompt = st.text_input("Ask a question about the document:")

if prompt:
    response = retrieval_chain.invoke({"input": prompt})
    st.write(response['answer'])
    
    with st.expander("Source Documents"):
        for doc in response['context']:
            st.write(doc.page_content)
            st.markdown("---")