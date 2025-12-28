import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
#from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# Load the Gemini API key
os.environ["GOOGLE_API_KEY"] = "xxxxx"
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key="xxxxx")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

loader = PyPDFLoader("MyDoc.pdf")  # Load your PDF file
data = loader.load()
#print(data)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=100)
docs = text_splitter.split_documents(data)

#print("Total number of Chunks: ", len(docs))  # Check how many chunks we have
#for chunk in docs:
 #   print(chunk.page_content)

# 3. Create FAISS Vector Store
# This creates the index in memory
vectorstore = FAISS.from_documents(documents=docs, embedding=embeddings)
#vectorstoredb = Chroma.from_documents(documents=docs, embedding=embeddings)
vectorstore.save_local("faiss_index_store")
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# 4. LCEL RAG Chain
#template = """Answer the question based only on the following context:
#{context}

#Question: {question}
#"""

#prompt = ChatPromptTemplate.from_template(template)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that answers questions based strictly on the provided context. If the answer is not in the context, say 'I cannot find the answer in the provided document.'"),
    ("human", "Here is the retrieved context: \n\n{context}\n\nQuestion: {question}")
])

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 5. Execute
print(rag_chain.invoke("what is JWI"))