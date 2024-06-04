import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set OpenAI API key from the environment variable
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Set the page layout to wide
st.set_page_config(layout="wide")

# Sidebar section
with st.sidebar:
   # Directory to store uploaded documents
   DOCS_DIR = os.path.abspath("/Users/suhaaspenugonda/RAG_Bot/uploaded_docs")
   
   # Create the directory if it doesn't exist
   if not os.path.exists(DOCS_DIR):
       os.makedirs(DOCS_DIR)
   
   st.subheader("Add to the Knowledge Base")
   
   # Form to upload files
   with st.form("my-form", clear_on_submit=True):
       uploaded_files = st.file_uploader("Upload a file to the Knowledge Base:", accept_multiple_files=True)
       submitted = st.form_submit_button("Upload!")

   # If files are uploaded and submitted
   if uploaded_files and submitted:
       for uploaded_file in uploaded_files:
           st.success(f"File {uploaded_file.name} uploaded successfully!")
           
           # Save the uploaded file to the document directory
           with open(os.path.join(DOCS_DIR, uploaded_file.name), "wb") as f:
               f.write(uploaded_file.read())

# Import necessary libraries and models
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI

# Initialize the language model and embedding model
llm = ChatOpenAI(model="gpt-3.5-turbo")
embedding = OpenAIEmbeddings(model="text-embedding-3-small")

# Import necessary components for processing and storing documents
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain.vectorstores import FAISS
import pickle

# Sidebar section for loading or creating a new vector store
with st.sidebar:
   use_existing_vector_store = st.radio("Use existing vector store if available", ["Yes", "No"], horizontal=True)

# Path to store the vector store
vector_store_path = "/Users/suhaaspenugonda/RAG_Bot/vectorstore.pkl"

# Load documents from the directory
raw_documents = DirectoryLoader(DOCS_DIR).load()

# Check if the vector store exists and load or create a new one
vector_store_exists = os.path.exists(vector_store_path)
vectorstore = None
if use_existing_vector_store == "Yes" and vector_store_exists:
   with open(vector_store_path, "rb") as f:
       vectorstore = pickle.load(f)
   with st.sidebar:
       st.success("Existing vector store loaded successfully.")
else:
   with st.sidebar:
       if raw_documents:
           with st.spinner("Splitting documents into chunks..."):
               # Split documents into smaller chunks
               text_splitter = CharacterTextSplitter(chunk_size=2500, chunk_overlap=300)
               chunks = text_splitter.split_documents(raw_documents)

           with st.spinner("Adding document chunks to vector database..."):
               # Create a new vector store from the document chunks
               vectorstore = FAISS.from_documents(chunks, embedding)

           with st.spinner("Saving vector store"):
               # Save the vector store to disk
               with open(vector_store_path, "wb") as f:
                   pickle.dump(vectorstore, f)
           st.success("Vector store created and saved.")
       else:
           st.warning("No documents available to process!", icon="⚠️")

# Chat section
st.subheader("Chat with my Assistant, RAG!")

# Initialize the chat messages if not present
if "messages" not in st.session_state:
   st.session_state.messages = []

# Display previous chat messages
for message in st.session_state.messages:
   with st.chat_message(message["role"]):
       st.markdown(message["content"])

# Import necessary components for prompting and parsing output
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate

# Define the chat prompt template
prompt_template = ChatPromptTemplate.from_messages(
   [("system", "You are a helpful AI assistant named RAG. You will reply to questions only based on the context that you are provided. If something is out of context, you will refrain from replying and politely decline to respond to the user."), ("user", "{input}")]
)

# Get user input
user_input = st.chat_input("Write a prompt")

# Set up the language model for generating responses
llm = ChatOpenAI(model="gpt-3.5-turbo")
chain = prompt_template | llm | StrOutputParser()

# If user input is provided and the vector store exists
if user_input and vectorstore is not None:
   # Add the user input to the chat messages
   st.session_state.messages.append({"role": "user", "content": user_input})
   
   # Retrieve relevant documents from the vector store
   docs = vectorstore.similarity_search_with_score(user_input, k=5)
   with st.chat_message("user"):
       st.markdown(user_input)

   # Prepare the context by concatenating relevant document contents
   context = ""
   for doc, _score in docs:
       context += doc.page_content + "\n\n"

   # Augment the user input with the context
   augmented_user_input = "Context: " + context + "\n\nQuestion: " + user_input + "\n"

   # Generate and display the response
   with st.chat_message("assistant"):
       message_placeholder = st.empty()
       full_response = ""

       for response in chain.stream({"input": augmented_user_input}):
           full_response += response
           message_placeholder.markdown(full_response + "▌")
       message_placeholder.markdown(full_response)
   
   # Add the assistant's response to the chat messages
   st.session_state.messages.append({"role": "assistant", "content": full_response})