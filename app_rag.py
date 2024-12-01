import os
from langchain_community.llms import Ollama
from dotenv import load_dotenv
from langchain_community.embeddings import OllamaEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
import streamlit as st
import tempfile

# Load environment variables
load_dotenv()

# Set up directories
persist_directory = "./rag_env/models/chroma_db"

# Initialize embeddings model
embed_model = OllamaEmbeddings(
    model="llama3.1",
    base_url='http://127.0.0.1:11434'
)

# Initialize the language model
llm = Ollama(model="llama3.1", base_url="http://127.0.0.1:11434")

# Streamlit app setup
st.title("Q&A Application with LangChain and Ollama")

# File uploader for PDF
uploaded_file = st.file_uploader("Load your PDF file :", type=["pdf"])

if uploaded_file is not None:
    # Save uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    # Load PDF content
    loader = PyPDFLoader(temp_file_path)
    documents = loader.load()

    # Split PDF content into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=128)
    chunks = text_splitter.split_documents(documents)

    # Create a Chroma vector store from the chunks
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embed_model,
        persist_directory=persist_directory
    )

    # Create a retriever from the vector store
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    # User input for questions
    question = st.text_input("Ask your question here :")

    # Handle the user's query
    if st.button("Get Answer"):
        if question:
            with st.spinner("Generating response..."):
                # First, try to get relevant documents
                relevant_docs = retriever.get_relevant_documents(question)
                
                # Check if we found any relevant documents
                if relevant_docs:
                    # Create context-specific prompt
                    context_prompt = f"""Based on the following context, please answer the question. 
                    If the question cannot be answered from the context, ignore the context completely 
                    and provide a general answer to the question instead.
                    
                    Question: {question}
                    
                    Context: {' '.join([doc.page_content for doc in relevant_docs])}
                    """
                    
                    response = llm.invoke(context_prompt)
                else:
                    # If no relevant documents found, use direct LLM response
                    response = llm.invoke(question)
                
                st.write(response)
        else:
            st.write("Please enter a question.")