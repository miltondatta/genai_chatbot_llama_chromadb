import os
import json

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
 
working_dir = os.path.dirname(os.path.abspath(__file__))

config_data = json.load(open(f"{working_dir}/config.json"))
GROQ_API_KEY = config_data["GROQ_API_KEY"]
os.environ["GROQ_API_KEY"] = GROQ_API_KEY


def setup_vectorstore():
    persist_dir = f"{working_dir}/vector_db_store"
    embeddings = HuggingFaceEmbeddings()
    vectorstore = Chroma(persist_directory=persist_dir,
                         embedding_function=embeddings)
    
    return vectorstore

def chat_chain(vectorstore):
    llm = ChatGroq(
        model="llama-3.1-70b-versatile",
        temperature=0.5,
        api_key=GROQ_API_KEY,
        max_tokens=4096,
        max_retries=2
    )
    
    retriever = vectorstore.as_retriever()
    
    memory = ConversationBufferMemory(
        llm=llm,
        output_key="answer",
        memory_key="chat_history",
        return_messages=True
    )
    
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        memory=memory,
        verbose=True,
        return_source_documents=True
    )
    
    return chain

    



