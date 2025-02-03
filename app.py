import json 
import os 
import sys 
import boto3
import streamlit as st

# Embeddings and AWS Bedrock
from langchain_core.embeddings import Embeddings
from langchain_aws.embeddings.bedrock import BedrockEmbeddings
from langchain_aws import ChatBedrockConverse
from langchain_aws import ChatBedrock

# Data Processing
import numpy as np 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

# Vector Store
from langchain_community.vectorstores import FAISS

# LLM Components
from langchain_core.prompts import ChatPromptTemplate
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
            client=bedrock,
            model_kwargs={
                "temperature": 0.7,
                "maxTokens": 2000,
                "stopSequences": []
            }
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
            client=bedrock,
            model_kwargs={
                "temperature": 0.7,
                "maxTokens": 2000,
                "stopSequences": []
            }
        )
        return llm
    except Exception as e:
        st.error(f"Error initializing Deepseek LLM: {str(e)}")
        return None

def get_response_llm(llm, vectorstore_faiss, query):
    try:
        docs = vectorstore_faiss.similarity_search(query, k=3)
        context = "\n".join(doc.page_content for doc in docs)
        
        body = json.dumps({
            "inferenceConfig": {
                "max_new_tokens": 1000
            },
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "text": f"Context: {context}\n\nQuestion: {query}\n\nProvide a technical analysis with examples and key components."
                        }
                    ]
                }
            ]
        }).encode('utf-8')
        
        response = bedrock.invoke_model(
            modelId=llm.model_id,
            contentType="application/json",
            accept="application/json",
            body=body
        )
        
        response_body = json.loads(response['body'].read())
        text = response_body['output']['message']['content'][0]['text']
        
        # Format text into streamlit components
        sections = text.split('###')
        formatted_response = ""
        for section in sections:
            if section.strip():
                parts = section.strip().split('\n', 1)
                if len(parts) > 1:
                    title, content = parts
                    formatted_response += f"## {title}\n\n"
                    formatted_response += content + "\n\n"

        return formatted_response

    except Exception as e:
        return f"Error: {str(e)}\nDocs retrieved: {len(docs) if 'docs' in locals() else 'None'}"

def main():
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
                   if docs and get_vector_store(docs):
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
                       
                       # Parse and display response nicely
                       if isinstance(response, dict):
                           content = response.get('output', {}).get('message', {}).get('content', [])
                           if content and isinstance(content[0], dict):
                               text = content[0].get('text', '')
                               
                               # Split text by sections and display with formatting
                               sections = text.split('###')
                               for section in sections:
                                   if section.strip():
                                       # Handle main sections
                                       parts = section.strip().split('\n', 1)
                                       if len(parts) > 1:
                                           title, content = parts
                                           st.subheader(title)
                                           
                                           # Handle subsections
                                           subsections = content.split('\n\n')
                                           for subsection in subsections:
                                               if subsection.strip():
                                                   if '**' in subsection:
                                                       st.markdown(subsection)
                                                   else:
                                                       st.write(subsection)
                                                       
                       else:
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
                       
                       # Use same response parsing logic
                       if isinstance(response, dict):
                           content = response.get('output', {}).get('message', {}).get('content', [])
                           if content and isinstance(content[0], dict):
                               text = content[0].get('text', '')
                               
                               sections = text.split('###')
                               for section in sections:
                                   if section.strip():
                                       parts = section.strip().split('\n', 1)
                                       if len(parts) > 1:
                                           title, content = parts
                                           st.subheader(title)
                                           subsections = content.split('\n\n')
                                           for subsection in subsections:
                                               if subsection.strip():
                                                   if '**' in subsection:
                                                       st.markdown(subsection)
                                                   else:
                                                       st.write(subsection)
                       else:
                           st.write(response)
                           
                       st.success("Done")
   else:
       st.info("Please enter a question above.")

if __name__ == "__main__":
   main()