import streamlit as st
import openai
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import os
from dotenv import load_dotenv

load_dotenv()

# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

## langchain Tracking 
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "end-end-chatbot"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

## prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant named ROME AI. Please response to the user queries"),
        ("user", "Question: {question}")
    ]
)

def generate_response(question, api_key,llm, temperature, max_tokens):
    openai.api_key = api_key
    llm = ChatOpenAI(model=llm)
    output_parser= StrOutputParser()
    chain=prompt|llm|output_parser
    answer = chain.invoke({"question": question})
    return answer

st.title("End-End Chatbot with OpenAI")
st.write("This is a simple chatbot that uses OpenAI to answer questions. Please ask a question and the chatbot will try to answer it.")
st.sidebar.title("Settings")
api_key=st.sidebar.text_input("Enter your Open AI API KEY!", type="password")

llm=st.sidebar.selectbox("Select the language model", ["gpt-3.5-turbo", "gpt-40", "gpt-4", "gpt-4-turbo"])

temperature=st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
max_tokens=st.sidebar.slider("Max Tokens", min_value=50, max_value=500, value=150, step=50)

st.write("Please ask a question")
user_input= st.text_input("You:")

if user_input and api_key:
    response= generate_response(user_input, api_key, llm, temperature, max_tokens)
    st.write(response)
elif user_input:
    st.write("Please enter your OpenAI API Key in the settings")
else:
    st.write("Please enter a question")
