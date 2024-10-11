import streamlit as st
import validators
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import (
    UnstructuredURLLoader,
    YoutubeLoader,
)
from langchain_groq import ChatGroq

## streamlit app

st.set_page_config(
    page_title="Langchain summarize text from Yotube or Website", page_icon="🐦"
)
st.title(" 🐦 Langchian : summarize text from YT or website")
st.subheader("Summarize URL")

with st.sidebar:
    groq_api_key = st.text_input("Groq API key", value="", type="password")


generic_url = st.text_input("URL", label_visibility="collapsed")

llm = ChatGroq(model="Gemma-7b-It", groq_api_key=groq_api_key)

prompt_template = """
Provide a summary of the following content in 300 words:
Content:{text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

if st.button("Summarize the Content from Youtube or Website"):
    ## valduate all inputs
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("please provide the information to get started")
    elif not validators.url(generic_url):
        st.error(
            "Please enter a valid url, It can nat be a YT video URl or website URL"
        )

    else:
        try:
            with st.spinner("Waiting..."):
                ## loading the werbsdites or yt video data
                if "youtube.com" in generic_url:
                    loader = YoutubeLoader.from_youtube_url(
                        generic_url, add_video_info=True
                    )
                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url], ssl_verify=False
                    )

                docs = loader.load()

                chain = load_summarize_chain(
                    llm, chain_type="stuff", prompt=prompt
                )
                output_summary = chain.run(docs)

                st.success(output_summary)
        except Exception as e:
            st.exception(f"Exception: {e}")
