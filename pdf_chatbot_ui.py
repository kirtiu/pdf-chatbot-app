# File: C:/Users/kirti.upadhyay/Documents/AI_Learning/pdf_chatbot_ui.py

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
#from langchain_chroma import Chroma
# NEW import
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import tempfile
import os


# ✅ Works both locally AND on cloud
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    from dotenv import load_dotenv
    load_dotenv()

# ════════════════════════════════════════
# PAGE CONFIG — must be first st command!
# ════════════════════════════════════════
st.set_page_config(
    page_title="PDF AI Chatbot",
    page_icon="📄",
    layout="centered"
)

# ════════════════════════════════════════
# TITLE & DESCRIPTION
# ════════════════════════════════════════
st.title("📄 PDF AI Chatbot")
st.caption("Upload a PDF and ask anything about it!")

# ════════════════════════════════════════
# SESSION STATE — persists across reruns
# ════════════════════════════════════════
if "messages" not in st.session_state:
    st.session_state.messages = []          # chat history

if "retriever" not in st.session_state:
    st.session_state.retriever = None       # RAG retriever

if "pdf_loaded" not in st.session_state:
    st.session_state.pdf_loaded = False     # PDF status

# ════════════════════════════════════════
# SIDEBAR — PDF Upload
# ════════════════════════════════════════
with st.sidebar:
    st.header("📁 Upload PDF")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type="pdf"
    )

    if uploaded_file and not st.session_state.pdf_loaded:
        with st.spinner("📖 Processing PDF..."):
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".pdf"
            ) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            # Load & chunk
            loader = PyPDFLoader(tmp_path)
            pages = loader.load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50
            )
            chunks = splitter.split_documents(pages)

            # Create vector store
            embeddings = OpenAIEmbeddings()

            vectorstore = FAISS.from_documents(
                documents=chunks,
                embedding=embeddings
)

            # Save retriever to session state
            st.session_state.retriever = vectorstore.as_retriever(
                search_kwargs={"k": 3}
            )
            st.session_state.pdf_loaded = True
            st.session_state.pdf_name = uploaded_file.name

            # Clean up temp file
            os.unlink(tmp_path)

        st.success(f"✅ {uploaded_file.name} loaded!")
        st.info(f"📊 {len(chunks)} chunks created")

    # Show status
    if st.session_state.pdf_loaded:
        st.success(f"📄 Active: {st.session_state.pdf_name}")

        # Reset button
        if st.button("🗑️ Clear & Upload New"):
            st.session_state.messages = []
            st.session_state.retriever = None
            st.session_state.pdf_loaded = False
            st.rerun()
    else:
        st.warning("⬆️ Please upload a PDF to start")

# ════════════════════════════════════════
# CHAT AREA — Display message history
# ════════════════════════════════════════
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# ════════════════════════════════════════
# WELCOME MESSAGE — shown before any chat
# ════════════════════════════════════════
if not st.session_state.messages:
    if st.session_state.pdf_loaded:
        with st.chat_message("assistant"):
            st.write(f"👋 Hi! I've loaded **{st.session_state.pdf_name}**. Ask me anything about it!")
    else:
        with st.chat_message("assistant"):
            st.write("👋 Hi! Please upload a PDF from the sidebar to get started.")

# ════════════════════════════════════════
# CHAT INPUT
# ════════════════════════════════════════
if prompt := st.chat_input(
    "Ask a question about your PDF...",
    disabled=not st.session_state.pdf_loaded
):
    # Add user message to history
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    # Show user message
    with st.chat_message("user"):
        st.write(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):

            # RAG — retrieve relevant chunks
            docs = st.session_state.retriever.invoke(prompt)
            context = "\n\n".join([d.page_content for d in docs])

            # Guardrail — check context
            if not context.strip():
                response = "I couldn't find relevant information in the PDF for that question."
            else:
                # Build prompt
                template = ChatPromptTemplate.from_messages([
                    ("system", """You are a helpful PDF assistant.
                    Answer questions based ONLY on the provided context.
                    If the answer isn't in the context, say so clearly.
                    Context: {context}"""),
                    ("human", "{question}")
                ])

                # LLM chain
                llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
                chain = template | llm | StrOutputParser()
                response = chain.invoke({
                    "context": context,
                    "question": prompt
                })

        st.write(response)

    # Save assistant response
    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })