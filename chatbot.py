import streamlit as st

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain.chains import ConversationalRetrievalChain

# Install dependencies (should be run separately in the terminal, not in the script)
# pip install langchain 
# pip install langchain-community 
# pip install streamlit langchain_core langchain_ollama pdfplumber
# pip install PyMuPDFLoader

template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""

# Load a text file 
file_path = "D:\\Uma\\Talk\\BIT-Generative-AI\\Code\\indian-penal-code-ncib.pdf"  # Update with the correct file path
loader = PDFPlumberLoader(file_path)
documents = loader.load()

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
text_chunks = text_splitter.split_documents(documents)

embeddings = OllamaEmbeddings(model="deepseek-r1:1.5b")
vector_store = InMemoryVectorStore(embeddings)
model = OllamaLLM(model="deepseek-r1:1.5b")
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
chatbot = ConversationalRetrievalChain.from_llm(model, retriever)

# User Chat Loop
print("🔹 IPC Chatbot Initialized! Ask about the Indian Penal Code.")
chat_history = []

while True:
    query = input("\n👤 You: ")
    if query.lower() in ["exit", "quit"]:
        print("🔹 Chatbot: Goodbye!")
        break
    
    response = chatbot.invoke({"question": query, "chat_history": chat_history})
    chat_history.append((query, response["answer"]))
    
    print(f"🔹 Chatbot: {response['answer']}")
