# RAG Q/A Bot

## Introduction
This repository contains a Retrieval-Augmented Generation (RAG) Question/Answer (Q/A) bot built using Streamlit, LangChain, and OpenAI's GPT-3.5-turbo model. This bot allows users to upload documents to build a knowledge base and then interact with the bot to retrieve information based on the content of the uploaded documents.

## Technologies Used
**Streamlit**: A framework for building interactive web applications in Python.

**LangChain**: A framework for building applications with language models.

**OpenAI GPT-3.5-turbo**: A language model used to generate responses.

**FAISS**: A library for efficient similarity search and clustering of dense vectors.

**Python dotenv**: A library to manage environment variables.

## Setup Instructions

**1. Environment Variables**
Ensure you have an .env file containing your OpenAI API key

OPENAI_API_KEY=your_openai_api_key

**2. Dependencies:**
Install the required Python packages using

pip install -r /path/.../libraries.txt

**3. Directory Structure:**

Create the directories to store uploaded documents and the vector store

## Application Workflow

### Sidebar
**1. Upload Documents**

Users can upload files to the knowledge base. These files are saved to the uploaded_docs directory.

**2. Use Existing Vector Store**

Users can choose to load an existing vector store or create a new one. The vector store is saved as vectorstore.pkl.

### Main Section

**1. Load Documents**

Documents are loaded from the uploaded_docs directory using DirectoryLoader.

**2. Vector Store Handling**

If an existing vector store is to be used and it exists, it is loaded from vectorstore.pkl.

Otherwise, documents are split into chunks using CharacterTextSplitter, and a new vector store is created and saved.

**3. Chat Interface**

Users can interact with the bot by inputting their questions. The bot retrieves relevant document chunks using FAISS and constructs a context for the language model.

The GPT-3.5-turbo model generates responses based on the provided context.

### Chat Messages

The chat messages are stored in the session state to maintain context during the conversation.

### Prompting and Response Generation

A custom prompt template ensures the bot responds only with information from the provided context. The ChatOpenAI model generates responses which are displayed in the chat interface.

## How to Run

**1. Start the Streamlit Application**

streamlit run /path/.../main.py

**2. Interact with the Bot**

Use the sidebar to upload documents and manage the vector store.

Enter questions in the chat interface to retrieve information based on the uploaded documents.
