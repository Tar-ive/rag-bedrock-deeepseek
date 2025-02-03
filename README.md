# PDF Chat Application with AWS Bedrock

A Streamlit-based application that enables users to chat with PDF documents using AWS Bedrock's language models. The application uses FAISS for efficient vector storage and retrieval, supporting both Nova and Deepseek language models for generating responses.

## Features

- PDF document ingestion and processing
- Vector embedding using Amazon Titan Embeddings
- Vector storage using FAISS
- Two LLM options:
  - Amazon Nova Lite
  - Deepseek
- Interactive Streamlit interface
- Context-aware responses with detailed explanations

## Prerequisites

- Python 3.x
- AWS Account with Bedrock access
- Configured AWS credentials

## Required Dependencies

```bash
pip install streamlit boto3 langchain-core langchain-aws langchain-text-splitters langchain-community numpy faiss-cpu pypdf
```

## Project Structure

```
.
├── data/           # Directory for PDF files
├── faiss_index/    # Generated vector store
└── app.py          # Main application file
```

## Setup

1. Clone the repository
2. Install dependencies
3. Configure AWS credentials
4. Create a `data` directory and add your PDF files
5. Run the application

## AWS Configuration

Ensure your AWS credentials are properly configured with access to Bedrock services. You can configure them using:

```bash
aws configure
```

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Update the vector store:
   - Click "Vectors Update" in the sidebar to process PDF files
   - Wait for confirmation of successful update

3. Ask questions:
   - Enter your question in the text input field
   - Choose between Nova or Deepseek models for response generation
   - View the generated response

## Key Components

### Vector Store Management
- `data_ingestion()`: Processes PDF documents from the data directory
- `get_vector_store()`: Creates and saves FAISS vector store
- `load_vector_store()`: Loads existing vector store

### LLM Integration
- `get_nova_llm()`: Initializes Amazon Nova Lite model
- `get_deepseek_llm()`: Initializes Deepseek model
- `get_response_llm()`: Generates responses using selected LLM

### Prompt Template
The application uses a specialized prompt template for generating detailed responses that include:
- Simple analogies
- Technical definitions
- Practical examples
- Key components breakdown

## Error Handling

The application includes comprehensive error handling for:
- Missing data directory
- Empty PDF documents
- Vector store creation/loading issues
- LLM initialization problems
- Query processing errors

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Notes

- Large PDF files may take longer to process during vector store updates
- Response generation time may vary based on the chosen LLM and query complexity
- Ensure sufficient AWS permissions for Bedrock service access