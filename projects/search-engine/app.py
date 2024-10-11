import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import AgentType, initialize_agent
from langchain.callbacks import StreamlitCallbackHandler
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
import os
from dotenv import load_dotenv
import chromadb.api

chromadb.api.client.SharedSystemClient.clear_system_cache()

load_dotenv()

# Initialize tools
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)
wikipedia_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wikipedia = WikipediaQueryRun(api_wrapper=wikipedia_wrapper)
search = DuckDuckGoSearchRun(name="Search")

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings()

# Initialize ChromaDB
vectorstore = Chroma(embedding_function=embeddings)

st.title("Langchain - Chat with search and embeddings")
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key", type="password")
openai_api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")

if 'messages' not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hello! I am a chatbot who can search the web and use embeddings. How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input(placeholder="enter your query..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    llm = ChatGroq(groq_api_key=api_key, model_name="Llama3-8b-8192", streaming=True)
    tools = [search, arxiv, wikipedia]
    
    # Create a text splitter for chunking
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    
    # Split the prompt into chunks
    chunks = text_splitter.split_text(prompt)
    
    # Add chunks to the vector store
    vectorstore.add_texts(chunks)
    
    # Perform a similarity search
    similar_docs = vectorstore.similarity_search(prompt, k=2)
    
    # Add similar documents to the context
    context = "\n".join([doc.page_content for doc in similar_docs])
    
    # Initialize the agent with the updated context
    search_agent = initialize_agent(
        tools, 
        llm, 
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
        handling_parsing_errors=True
    )
    
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = search_agent.run(st.session_state.messages + [{"role": "system", "content": f"Additional context: {context}"}], callbacks=[st_cb])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)

# Persist the vector store
vectorstore.persist()