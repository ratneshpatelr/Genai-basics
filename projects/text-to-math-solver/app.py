import streamlit as st
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chains import LLMChain, LLMMathChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_groq import ChatGroq

st.set_page_config(page_title="Text to math solver")
st.title("Text to Math solver")

groq_api_key=st.sidebar.text_input(label="Groq API KEy", type="password")

if not groq_api_key:
    st.info("please add your Groq api key")
    st.stop()

llm = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)

wikipedia_wrapper = WikipediaAPIWrapper()
wikipedia_tool = Tool(
    name= "Wikipedia",
    func=wikipedia_wrapper.run,
    description="A tool for searching the internet to find the various infornamtion on topcis mentioned"
)

math_chain = LLMMathChain.from_llm(llm=llm)
calculator = Tool(
    name="Calculator",
    func=math_chain.run,
    description="A tools for answetring math related questions. Only input mathematical expression need to be provided"
)


prompt="""
You are a agent tasked for solving users mathematical questions. Logically arrive at the solution and provide a detailed explanation and display it point wise for the question below
Question:{question}
Answer:
"""
prompt_template=PromptTemplate(
    input_variable=["question"],
    template=prompt
)

chain=LLMChain(llm=llm, prompt=prompt_template)

reasoning_tool = Tool(
    name="Reasoning tool",
    func=chain.run,
    description="A tool for answering logic based and reasoning questions"
)

## Intiitalzie the agents

assistant_agent=initialize_agent(
    tools=[wikipedia_tool, calculator, reasoning_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True
)

if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {"role": "assistant", "content":"Hi, I am a Math chatbot who can answer all your math questions"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

question=st.text_area("Enter your question", "I have 5 bananas and 7 grapes , I eat 2 bananas and give away 3 grapes. Then I buy a dozen of apples and 2 packs of berries. Each pack of berries contains 25 berries, How many total pieces of fruit do i have at the end")

if st.button("find my answer"):
    if question:
        with st.spinner("Generate response"):
            st.session_state.messages.append({"role": "user","content": question})
            st.chat_message("user").write(question)

            st_cb=StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = assistant_agent.run(st.session_state.messages, callbacks=[st_cb])

            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write("Response")
            st.success(response)

    else:
        st.warning("please enter the question")
