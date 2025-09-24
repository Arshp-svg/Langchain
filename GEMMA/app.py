import os
import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from  langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

##Loading Environment Variables
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

st.title("Document Q&A with Groq and GEMMA")
llm=ChatGroq(groq_api_key=groq_api_key, model="gemma2-9b-it")

prompt=ChatPromptTemplate.from_template(
    """answer the questions based on the provided context only.
    please provide the most relevant answer based on the context.
    <context>
    {context}
    </context>
    Question: {input}"""
                                       
)
def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")  # using Google embedding model
        st.session_state.loader = PyPDFDirectoryLoader("./us_sensus", glob="*.pdf")  # load PDFs
        st.session_state.documents = st.session_state.loader.load()  # read documents
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # split into chunks

        # Split and limit the number of chunks (TEMP FIX to avoid quota issues)
        all_chunks = st.session_state.text_splitter.split_documents(st.session_state.documents)
        st.session_state.chunks = all_chunks[:20]  # Only use first 10 chunks to stay within quota

        # Create vector store from limited chunks
        st.session_state.vectors = FAISS.from_documents(
            st.session_state.chunks,
            embedding=st.session_state.embeddings
        )

        
input_prompt=st.text_input("Enter your question from documents")
if st.button("Embed Documents"):
    vector_embedding()
    st.write("Documents embedded successfully! You can now ask questions.")

if input_prompt:
    if "vectors" not in st.session_state:
        st.write("Please embed documents first.")
    else:
        chain=create_stuff_documents_chain(llm=llm, prompt=prompt)
        retriver=st.session_state.vectors.as_retriever()
        doc_chain=create_retrieval_chain(retriever=retriver, combine_docs_chain=chain)
        response=doc_chain.invoke({"input": input_prompt}) 
        st.write(response['answer'])
        
        with st.expander("Source Documents"):
            for i, source in enumerate(response['context']):
                st.write(source.page_content)
                st.write("---")
        