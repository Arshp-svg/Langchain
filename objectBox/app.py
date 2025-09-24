import streamlit as st
import os
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_objectbox.vectorstores import ObjectBox
from langchain_community.document_loaders import PyPDFDirectoryLoader

load_dotenv()

os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")
groq_api_key = os.getenv("GROQ_API_KEY")

st.title("Document Q&A with ObjectBox and LLaMA 3")
st.write("Upload your documents and ask questions about their content.")

llm=ChatGroq(model="llama-3.3-70b-versatile",api_key=groq_api_key)

promt=ChatPromptTemplate.from_template(
    """answer the questions based on the provided context only.
    please provide the most relevant answer based on the context.
    <context>
    {context}
    </context>
    Question: {input}"""
                                       
)

def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings= HuggingFaceEmbeddings()
        st.session_state.loader= PyPDFDirectoryLoader("./us_sensus", glob="*.pdf") ##data ingestion
        st.session_state.documents= st.session_state.loader.load() #loading documents
        st.session_state.text_splitter= RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200) #splitting documents into chunks
        st.session_state.chunks= st.session_state.text_splitter.split_documents(st.session_state.documents[:20]) #splitting documents
        st.session_state.vectors= ObjectBox.from_documents(st.session_state.chunks, embedding=st.session_state.embeddings,embedding_dimensions=768) #creating vector store
        

input_prompt=st.text_input("Enter your question from documents")

if st.button("Embed Documents"):
    vector_embedding()
    st.write("Documents embedded successfully! You can now ask questions.")

if input_prompt:
    if "vectors" not in st.session_state:
        st.warning("Please embed documents first by clicking the 'Embed Documents' button.")
    else:
        chain=create_stuff_documents_chain(llm=llm, prompt=promt)
        retriver=st.session_state.vectors.as_retriever()
        doc_chain=create_retrieval_chain(retriever=retriver, combine_docs_chain=chain)
        start=time.process_time()
        response=doc_chain.invoke({"input": input_prompt}) 
        st.write("Response time:", time.process_time()-start)
        st.write(response['answer'])
        
        with st.expander("Source Documents"):
            for i, source in enumerate(response['context']):
                st.write(source.page_content)
                st.write("---")
                
                
        