from fastapi import FastAPI
from langserve import add_routes
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
import os
import uvicorn
from dotenv import load_dotenv

load_dotenv()

load_dotenv()

os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")


app=FastAPI(
    title="LangChain API",
    description="API for LangChain",
    version="0.1.0"
)

llm=OllamaLLM(model="gemma")
add_routes(
    app,
    ChatOpenAI(),
    path="/gemma"
)



prompt1=ChatPromptTemplate.from_template("What is a good name for a company that makes {product}?")
prompt2=ChatPromptTemplate.from_template("What is a good slogan for a company that makes {product}?")

add_routes(
    app,
    prompt1 |llm,
    path="/name"
    )

add_routes(
    app,
    prompt2 |llm,
    path="/slogan"
    )


if __name__=="__main__":
    uvicorn.run(app, host="localhost", port=8000)