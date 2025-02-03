import json 
import os 
import sys 
import boto3
import streamlit as st

# Embeddings and AWS Bedrock
from langchain_core.embeddings import Embeddings
from langchain_aws.embeddings.bedrock import BedrockEmbeddings
from langchain_aws.chat_models.bedrock import convert_messages_to_prompt_mistral
from langchain_aws import ChatBedrock

# Data Processing
import numpy as np 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

# Vector Store
from langchain_community.vectorstores import FAISS

# LLM Components
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Initialize Bedrock client
bedrock = boto3.client(service_name='bedrock-runtime')
bedrock_embeddings = BedrockEmbeddings(
    model_id='amazon.titan-embed-text-v2:0',
    client=bedrock
)

def data_ingestion():
    """
    Load and process PDF documents from the data directory.
    Returns processed document chunks.
    """
    try:
        if not os.path.exists('data'):
            raise Exception("Data directory not found")
        
        loader = PyPDFDirectoryLoader('data')
        documents = loader.load()
        
        if not documents:
            raise Exception("No PDF documents found in data directory")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=10000,
            chunk_overlap=1000
        )
        docs = text_splitter.split_documents(documents)
        return docs
    except Exception as e:
        st.error(f"Error in data ingestion: {str(e)}")
        return None

def get_vector_store(docs):
    """
    Create and save FAISS vector store from documents.
    """
    try:
        if docs is None:
            raise Exception("No documents provided")
            
        vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
        vectorstore_faiss.save_local('faiss_index')
        return vectorstore_faiss
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return None

def load_vector_store():
    """
    Load existing FAISS vector store.
    """
    try:
        return FAISS.load_local(
            "faiss_index", 
            bedrock_embeddings,
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        st.error(f"Error loading vector store: {str(e)}")
        return None

def get_nova_llm():
    """
    Initialize Nova LLM model.
    """
    try:
        llm = ChatBedrock(
            model_id='amazon.nova-lite-v1:0',
            client=bedrock
        )
        return llm
    except Exception as e:
        st.error(f"Error initializing Nova LLM: {str(e)}")
        return None

def get_deepseek_llm():
    """
    Initialize Deepseek LLM model.
    """
    try:
        llm = ChatBedrock(
            model_id='deepseek-llm-r1',
            client=bedrock
        )
        return llm
    except Exception as e:
        st.error(f"Error initializing Deepseek LLM: {str(e)}")
        return None

from langchain_core.prompts import ChatPromptTemplate

# Define this at the top of your file with other imports
template = ChatPromptTemplate.from_messages([
    ("system", """You are an expert technical explainer. Analyze the provided context and provide detailed answers."""),
    ("human", """Context: {context}
    
    Question: {question}
    
    Please provide a detailed answer that includes:
    - Simple analogies from everyday life
    - Clear technical definitions
    - Practical examples
    - Key components breakdown
    
    If you don't have enough information, please say so.""")
])

def get_response_llm(llm, vectorstore_faiss, query):
    """
    Get response from LLM using the vector store.
    """
    try:
        if not query.strip():
            raise ValueError("Empty query provided")
            
        # Get relevant documents
        docs = vectorstore_faiss.similarity_search(query, k=3)
        
        if not docs:
            return "No relevant documents found to answer this question."
            
        # Combine the content from all documents
        context = "\n".join([doc.page_content for doc in docs])
        
        # Format messages using the template
        formatted_messages = template.format_messages(
            context=context,
            question=query
        )
        
        # Use invoke to get response
        response = llm.invoke(formatted_messages)
        
        # Return the content from the response
        return response.content

    except Exception as e:
        error_message = f"Error getting response: {str(e)}\n"
        error_message += "Debug info: "
        error_message += f"\nQuery: {query}"
        error_message += f"\nDocs retrieved: {len(docs) if 'docs' in locals() else 'None'}"
        st.error(error_message)
        return error_message
    
def main():
    """
    Main Streamlit application.
    """
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using AWS Bedrock💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    # Sidebar for vector store updates
    with st.sidebar:
        st.title("Update Or Create Vector Store:")
        
        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                try:
                    docs = data_ingestion()
                    if docs:
                        if get_vector_store(docs):
                            st.success("Vector store updated successfully!")
                        else:
                            st.error("Failed to create vector store")
                except Exception as e:
                    st.error(f"Error updating vectors: {str(e)}")

    # Only show model buttons if there's a question
    if user_question:
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Nova Output"):
                with st.spinner("Processing..."):
                    faiss_index = load_vector_store()
                    if faiss_index is None:
                        st.error("Failed to load vector store. Please update vectors first.")
                        return
                        
                    llm = get_nova_llm()
                    if llm:
                        response = get_response_llm(llm, faiss_index, user_question)
                        st.write(response)
                        st.success("Done")
        
        with col2:
            if st.button("Deepseek Output"):
                with st.spinner("Processing..."):
                    faiss_index = load_vector_store()
                    if faiss_index is None:
                        st.error("Failed to load vector store. Please update vectors first.")
                        return
                        
                    llm = get_deepseek_llm()
                    if llm:
                        response = get_response_llm(llm, faiss_index, user_question)
                        st.write(response)
                        st.success("Done")
    else:
        st.info("Please enter a question above.")

if __name__ == "__main__":
    main()