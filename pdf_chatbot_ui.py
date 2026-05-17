# File: pdf_chatbot_ui.py (FULL UPDATED VERSION)

import streamlit as st
import os
import time

# ── API KEYS ──────────────────────────────────────────────
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    from dotenv import load_dotenv
    load_dotenv()

# ── LANGSMITH TRACING ─────────────────────────────────────
if "LANGCHAIN_API_KEY" in st.secrets:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
    os.environ["LANGCHAIN_PROJECT"] = st.secrets.get(
        "LANGCHAIN_PROJECT", "pdf-chatbot"
    )

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import tempfile

# ── PAGE CONFIG ───────────────────────────────────────────
st.set_page_config(
    page_title="PDF AI Chatbot",
    page_icon="📄",
    layout="centered"
)

st.title("📄 PDF AI Chatbot")
st.caption("Upload a PDF and ask anything about it!")

# ── SESSION STATE ─────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "pdf_loaded" not in st.session_state:
    st.session_state.pdf_loaded = False
if "metrics" not in st.session_state:
    st.session_state.metrics = {
        "total_questions": 0,
        "total_response_time": 0.0,
        "questions_log": []
    }

# ── SIDEBAR ───────────────────────────────────────────────
with st.sidebar:
    st.header("📁 Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file and not st.session_state.pdf_loaded:
        with st.spinner("📖 Processing PDF..."):
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".pdf"
            ) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            loader = PyPDFLoader(tmp_path)
            pages = loader.load()
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500, chunk_overlap=50
            )
            chunks = splitter.split_documents(pages)
            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.from_documents(
                documents=chunks,
                embedding=embeddings
            )
            st.session_state.retriever = vectorstore.as_retriever(
                search_kwargs={"k": 3}
            )
            st.session_state.pdf_loaded = True
            st.session_state.pdf_name = uploaded_file.name
            os.unlink(tmp_path)

        st.success(f"✅ {uploaded_file.name} loaded!")
        st.info(f"📊 {len(chunks)} chunks created")

    if st.session_state.pdf_loaded:
        st.success(f"📄 Active: {st.session_state.pdf_name}")
        if st.button("🗑️ Clear & Upload New"):
            st.session_state.messages = []
            st.session_state.retriever = None
            st.session_state.pdf_loaded = False
            st.session_state.metrics = {
                "total_questions": 0,
                "total_response_time": 0.0,
                "questions_log": []
            }
            st.rerun()
    else:
        st.warning("⬆️ Please upload a PDF to start")

    # ── METRICS DASHBOARD ─────────────────────────────────
    if st.session_state.metrics["total_questions"] > 0:
        st.divider()
        st.header("📊 Session Metrics")
        m = st.session_state.metrics
        total_q = m["total_questions"]
        avg_time = round(m["total_response_time"] / total_q, 2)

        col1, col2 = st.columns(2)
        col1.metric("❓ Questions", total_q)
        col2.metric("⚡ Avg Time", f"{avg_time}s")

        st.subheader("📋 Recent Questions")
        for i, log in enumerate(reversed(m["questions_log"][-5:])):
            st.caption(
                f"• {log['question']}... "
                f"({log['time']}s)"
            )

# ── CHAT DISPLAY ──────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if not st.session_state.messages:
    if st.session_state.pdf_loaded:
        with st.chat_message("assistant"):
            st.write(
                f"👋 Hi! I've loaded **{st.session_state.pdf_name}**."
                " Ask me anything about it!"
            )
    else:
        with st.chat_message("assistant"):
            st.write("👋 Hi! Please upload a PDF from the sidebar.")

# ── CHAT INPUT ────────────────────────────────────────────
if prompt := st.chat_input(
    "Ask a question about your PDF...",
    disabled=not st.session_state.pdf_loaded
):
    st.session_state.messages.append({
        "role": "user", "content": prompt
    })
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):

            start_time = time.time()

            docs = st.session_state.retriever.invoke(prompt)
            context = "\n\n".join([d.page_content for d in docs])

            if not context.strip():
                response = "I couldn't find relevant information in the PDF."
            else:
                template = ChatPromptTemplate.from_messages([
                    ("system", """You are a helpful PDF assistant.
                    Answer ONLY from context. Say so if not found.
                    Context: {context}"""),
                    ("human", "{question}")
                ])
                llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
                chain = template | llm | StrOutputParser()
                response = chain.invoke({
                    "context": context,
                    "question": prompt
                })

            elapsed = round(time.time() - start_time, 2)

            # Update metrics
            st.session_state.metrics["total_questions"] += 1
            st.session_state.metrics["total_response_time"] += elapsed
            st.session_state.metrics["questions_log"].append({
                "question": prompt[:50],
                "time": elapsed,
                "sources": len(docs)
            })

        st.write(response)

    st.session_state.messages.append({
        "role": "assistant", "content": response
    })