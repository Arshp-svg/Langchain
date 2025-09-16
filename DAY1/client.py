import requests
import streamlit as st


def get_name_response(prompt: str):
    response = requests.post(
        "http://localhost:8000/name/invoke", json={"input": {"product": prompt}}
    )
    return response.json()["output"]


def get_slogan_response(prompt: str):
    response = requests.post(
        "http://localhost:8000/slogan/invoke", json={"input": {"product": prompt}}
    )
    return response.json()["output"]


st.title("Gemma LLM with LangServe")
user_input1 = st.text_input("Enter a product description:")
user_input2 = st.text_input("Enter a product description for slogan:")

if user_input1:
    output = get_name_response(user_input1)
    st.write("Response:", output)
    st.success("Request completed successfully!")

if user_input2:
    output = get_slogan_response(user_input2)
    st.write("Response:", output)
    st.success("Request completed successfully!")

