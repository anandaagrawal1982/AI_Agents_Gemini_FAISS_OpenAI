import streamlit as st
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- Page Config ---
st.set_page_config(page_title="Gemini PDF Chat", layout="wide")
st.title("📄 Chat with your PDF (Gemini + FAISS)")

# --- Sidebar: Setup & File Upload ---
with st.sidebar:
    #api_key = st.text_input("Enter Gemini API Key", type="password")
    #uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    process_button = st.button("Process Document")

# --- Initialize Session State ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# --- Step 1: Document Processing ---
if process_button:
    with st.spinner("Analyzing PDF..."):
        # Save temp file to load it
        #with open("temp.pdf", "wb") as f:
        #    f.write(uploaded_file.getvalue())
        
        loader = PyPDFLoader("MyDoc.pdf")
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        splits = text_splitter.split_documents(docs)
        
        os.environ["GOOGLE_API_KEY"] = "xxxxx"
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        #os.environ["GOOGLE_API_KEY"] = api_key
        #embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.session_state.vector_store = FAISS.from_documents(splits, embeddings)
        st.success("PDF Processed! You can now ask questions.")

# --- Step 2: Chat Interface ---
# Display previous messages
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if user_query := st.chat_input("Ask something about the PDF..."):
    if not st.session_state.vector_store:
        st.error("Please upload and process a PDF first!")
    else:
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # --- LCEL RAG Chain Execution ---
        with st.chat_message("assistant"):
            os.environ["GOOGLE_API_KEY"] = "xxxxx"
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
            retriever = st.session_state.vector_store.as_retriever()
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", "Use the context to answer the user. If unknown, say so."),
                ("human", "Context: {context}\n\nQuestion: {question}")
            ])

            rag_chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt | llm | StrOutputParser()
            )

            response = rag_chain.invoke(user_query)
            st.markdown(response)
            st.session_state.chat_history.append({"role": "assistant", "content": response})