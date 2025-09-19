import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retrieval import create_rag_chain
from src.knowledge_graph import knowledge_graph_page
from src.ingestion_page import ingestion_page
from langchain_chroma import Chroma
from src import config

st.set_page_config(page_title="Legal Research Engine", layout="wide")

PAGES = {
    "Chat": "chat_page",
    "Knowledge Graph": "knowledge_graph_page",
    "Data Ingestion": "ingestion_page"
}

st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", list(PAGES.keys()))

@st.cache_resource
def load_rag_chain():
    return create_rag_chain()

import logging

logging.basicConfig(level=logging.INFO)

def load_vectorstore():
    try:
        from langchain_openai import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings(
            model=config.EMBEDDING_MODEL,
            openai_api_base="http://127.0.0.1:1234/v1",
            openai_api_key="lm-studio",
            check_embedding_ctx_length=False
        )
        vectorstore = Chroma(
            persist_directory=config.CHROMA_PATH,
            embedding_function=embeddings,
            collection_metadata=config.CHROMA_CONFIG
        )
        logging.info("Vector store loaded successfully.")
        return vectorstore
    except Exception as e:
        logging.error(f"Error loading vector store: {e}")
        st.error(f"Error loading vector store: {e}")
        return None

rag_chain = load_rag_chain()
vectorstore = load_vectorstore()

def chat_page():
    st.title("Legal Research Engine")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                with st.expander("Sources"):
                    for source in message["sources"]:
                        st.write(f"**Source:** {source.metadata.get('source', 'Unknown')}")
                        st.write(f"**Page:** {source.metadata.get('page', 'Unknown')}")
                        st.write(f"**Content:** {source.page_content}")

    if prompt := st.chat_input("What is your legal question?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = rag_chain(prompt)
                st.markdown(response["answer"])
                
                sources = response.get("context", [])
                if sources:
                    with st.expander("Sources"):
                        for source in sources:
                            st.write(f"**Source:** {source.metadata.get('source', 'Unknown')}")
                            st.write(f"**Page:** {source.metadata.get('page', 'Unknown')}")
                            st.write(f"**Content:** {source.page_content}")

                message = {"role": "assistant", "content": response["answer"]}
                if sources:
                    message["sources"] = sources
                st.session_state.messages.append(message)

if vectorstore:
    if selection == "Chat":
        logging.info("Loading chat page.")
        chat_page()
    elif selection == "Knowledge Graph":
        logging.info("Loading knowledge graph page.")
        knowledge_graph_page(vectorstore)
    elif selection == "Data Ingestion":
        logging.info("Loading data ingestion page.")
        ingestion_page()
else:
    st.error("Vector store could not be loaded. Please check the logs for more information.")