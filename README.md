# PDF Chatbot with RAG
A Streamlit-based PDF chatbot application that uses Retrieval-Augmented Generation (RAG) to answer questions from uploaded PDF documents. The application processes PDF files, creates vector embeddings, and provides intelligent responses using the DeepSeek language model.

# App link: <a href="https://llm-rag-ypcbpwsrxnypbibdfmvert.streamlit.app/" target="_blank">Open the RAG LLM App</a>

# Features
Multi-PDF Support: Upload and process multiple PDF files simultaneously

Intelligent Query Processing: Uses RAG architecture for accurate, context-aware responses

Vector Database: Utilizes Pinecone for efficient document retrieval

Advanced Language Model: Powered by DeepSeek-R1-Distill-Qwen-32B for high-quality responses

User-Friendly Interface: Clean Streamlit interface with real-time processing feedback

# How It Works
## Document Processing:

Upload PDF files through the sidebar

Documents are split into chunks using recursive text splitting

Text is processed and prepared for embedding

## Vector Storage:

Uses SentenceTransformer embeddings (all-MiniLM-L6-v2)

Stores document vectors in Pinecone vector database

Creates a retriever for efficient similarity search

## Question Answering:

Processes user queries with minimum 3-word requirement

Retrieves relevant context from the vector store

Generates accurate answers using the DeepSeek language model

Provides fallback responses when answers aren't in context

# Installation
bash
pip install numpy pandas streamlit PyGithub pinecone-client langchain-core pymupdf unidecode langchain-community sentence-transformers langchain-huggingface langchain-pinecone

# Usage
Run the application:

bash
streamlit run app.py
Upload PDF files using the sidebar file uploader

Click "Submit & Process" to process the documents

Enter your query in the text input (minimum 3 words)

Click "Submit" to get the answer

# Technical Architecture

Frontend: Streamlit

Vector Database: Pinecone

Embeddings: SentenceTransformer (all-MiniLM-L6-v2)

LLM: DeepSeek-R1-Distill-Qwen-32B via HuggingFace

Text Processing: PyMuPDF, LangChain text splitters

RAG Framework: LangChain

# Configuration

The application requires the following API keys:

Pinecone API key

HuggingFace API token

# Author
Vikrant Nikunj - Data Science Enthusiast

# Note
Ensure you have sufficient API credits for Pinecone and HuggingFace

Processing time may vary based on document size and complexity

The application requires a minimum of 3 words for queries to ensure meaningful responses
