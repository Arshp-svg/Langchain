import os
import streamlit as st
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Load env variables
load_dotenv()
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Streamlit UI
st.title("🧠 Chat with Llama 2 (Free via Ollama)")

user_input = st.text_input("Ask me anything:")

prompt = ChatPromptTemplate.from_messages([
    ("system", "you are a helpful assistant that answers the user queries."),
    ("user", "{question}")
])

llm = OllamaLLM(model="gemma")
output = StrOutputParser()
chain = prompt | llm | output

if user_input:
    response = chain.invoke({"question": user_input})
    st.text_area("Response:", value=response, height=200)
