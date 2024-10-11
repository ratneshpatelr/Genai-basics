## Rag Q&A conversation with PDF including Chat History

import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma # vector store database
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.chat_message_histories import ChatMessageHistory
import os
from dotenv import load_dotenv
import chromadb.api

chromadb.api.client.SharedSystemClient.clear_system_cache()

load_dotenv()

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

st.title("Rag Q&A conversation with PDF including Chat History")
st.write("Upload PDF's and chat withg their content")

api_key= st.text_input("Enter your GROQ API Key", type="password")

if api_key:
    llm = ChatGroq(groq_api_key=api_key, model_name="Gemma2-9b-It")

    session_id = st.text_input("Enter your session ID", value="default_session")

    if "store" not in st.session_state:
        st.session_state.store = {}

    uploaded_files = st.file_uploader("Choose a PDF file", type=["pdf"], accept_multiple_files=True)

    if uploaded_files:
        documents= []
        for upload_fle in uploaded_files:
            tempPdf = f"./temp.pdf"
            with open(tempPdf, "wb") as file:
                file.write(upload_fle.getvalue())
                file_name=upload_fle.name
            
            loader = PyPDFLoader(tempPdf)
            docs= loader.load()
            documents.extend(docs) # data ingestion 
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = text_splitter.split_documents(documents)
        vector_store = Chroma.from_documents(documents=splits, embedding=embeddings)
        retriever = vector_store.as_retriever()

        contextualize_q_system_prompt= (
            "Given a chat history and latest user question"
            "which might reference context in chat history"
            "formulate a standlone question which can be understood"
            "without the chat history. DO NOT answer the question"
            "just reformulate it if needed and otheriwse return it as is."
        )
        contextualize_q_prompt= ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )
        history_aware_retreiver=create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

        system_prompt = (
            "You are an assistant for question-asnwering tasks"
            "Use the following pieces of retrieved context to answer"
            "the question. If you don't knowe the answer, say that you"
            "don't know. Use three sentances maximum and keep the"
            "answer concise"
            "\n\n"
            "{context}"
        )
        qa_prompt= ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )

        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retreiver, question_answer_chain)

        def get_session_history(session: str) -> BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]

        conversational_rag_chain=RunnableWithMessageHistory(
            runnable=rag_chain,
            get_session_history=get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

        user_input = st.text_input("Ask a question")

        if user_input:
            session_history = get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={
                    "configurable": {"session_id": session_id},
                }
            )
            st.write(st.session_state.store)    
            st.write(response["answer"])
            st.write("Chat History", session_history.messages)


else:
    st.warning("Please enter your GROQ API Key")


