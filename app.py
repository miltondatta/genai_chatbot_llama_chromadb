
import streamlit as st
from main import chat_chain,setup_vectorstore

st.set_page_config(
    page_title="Medical Chat Bot",
    page_icon="🫀",
    layout="centered"
)

st.title("🫀 Multi Documents Medical Chat Bot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = setup_vectorstore()

if "conversational_chain" not in st.session_state:
    st.session_state.conversational_chain = chat_chain(st.session_state.vectorstore)

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask AI......")
if user_input:
    st.session_state.chat_history.append({"role":"user","content":user_input})
    
    with st.chat_message("user"):
        st.markdown(user_input)
    
    with st.chat_message("assistant"):
        response = st.session_state.conversational_chain({"question":user_input})
        assistant_response =response["answer"]
        st.markdown(assistant_response)
        st.session_state.chat_history.append({"role":"assistant","content":assistant_response})
    

