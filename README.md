# Multilingual RAG Chatbot

![Chatbot Icon](static/CatRag.ico)

This is a Multilingual Retrieval-Augmented Generation (RAG) Chatbot built using **LangChain** and **Streamlit**. It allows users to upload their own documents (PDF, TXT, DOCX) and ask questions about them in multiple languages. The chatbot leverages **FAISS** for efficient vector database indexing and retrieval, and supports various Large Language Models (LLMs) including OpenAI, Google (Gemini), Anthropic (Claude), and local Ollama models.

## Features

-   **Multilingual Support**: Supports queries and documents in multiple languages.
-   **Flexible LLM Integration**: Connects with OpenAI, Google Gemini, Anthropic Claude, and local Ollama models.
-   **Document Upload**: Allows users to upload PDF, TXT, and DOCX files.
-   **Dynamic Ingestion**: Uploaded documents are embedded and added to a FAISS vector store dynamically.
-   **Metadata Tracking**: Maintains metadata for ingested documents, including file name, word count, and language detection.
-   **Efficient Retrieval**: Utilizes FAISS for fast and accurate document retrieval.
-   **Streamlit UI**: Intuitive and interactive user interface for seamless interaction.
-   **Logging**: Comprehensive logging to `logs/rag_chatbot.log` and console for monitoring and debugging.

## Technologies Used

This project extensively uses the following key libraries and frameworks:

-   **LangChain**: The core framework for building LLM-powered applications. LangChain is used for:
    -   **Document Loaders**: To load various document types (PDF, TXT, DOCX) into a standardized format.
    -   **Text Splitters**: To break down large documents into smaller, manageable chunks suitable for embedding and retrieval.
    -   **Embeddings**: To convert text chunks into numerical vector representations using multilingual sentence transformers (e.g., `all-MiniLM-L6-v2`).
    -   **Vector Stores**: Integration with FAISS for efficient storage and similarity search of document embeddings.
    -   **Retrievers**: To fetch the most relevant document chunks based on user queries.
    -   **Chains**: To orchestrate the RAG process, combining document retrieval with LLM generation to answer questions based on the retrieved context.

-   **Streamlit**: Used for building the interactive and user-friendly web interface. Streamlit simplifies the creation of data apps with its component-based structure, allowing for quick development of the chatbot's UI, including file uploaders, model selectors, and chat history display.

-   **FAISS (Facebook AI Similarity Search)**: An open-source library for efficient similarity search and clustering of dense vectors. In this project, FAISS serves as the vector database to store and quickly retrieve document embeddings, enabling fast context retrieval for the RAG process.

-   **Hugging Face Transformers & Sentence Transformers**: Utilized for generating high-quality, multilingual embeddings for the document chunks, ensuring the chatbot can understand and process information across different languages.

-   **Python-dotenv**: For managing environment variables, allowing secure handling of API keys and other sensitive configurations.

-   **Langdetect**: A language detection library used to identify the language of uploaded documents and their chunks, supporting the multilingual capabilities of the chatbot.

## Project Structure

├── .env                     # Environment variables (optional)
├── src/
│   ├── init.py
│   ├── config.py            # Configuration settings
│   ├── generator.py         # Handles LLM generation using selected model
│   ├── ingest.py            # Loads, processes, and embeds uploaded documents
│   ├── interface.py         # Streamlit app logic
│   ├── retriever.py         # Handles FAISS search and document retrieval
│   ├── logger_setup.py      # Logging configuration
├── embeddings/
│   ├── documents.json       # Document metadata
│   ├── faiss_index/         # FAISS vector database
├── how_to_run               # Detailed instructions on how to run the application
├── main.py                  # Runs the Streamlit app
├── requirements.txt         # Python dependencies
├── README.md                # Project README file

## Installation

1.  **Clone the repository (or download the project files):**
    ```bash
    git clone https://github.com/ninaddhatrak/Multilingual-RAG-Chatbot.git
    cd multilingual-rag-chatbot
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```


## Docker Deployment

1. Build the Docker image:
   ```
   docker build -t multilingual-rag-chatbot .
   ```

2. Run the container:
   ```
   docker run -p 8501:8501 multilingual-rag-chatbot
   ```

3. (Optional) Run with persistent storage for embeddings and logs:
   ```
   docker run -p 8501:8501 -v $(pwd)/embeddings:/app/embeddings -v $(pwd)/logs:/app/logs multilingual-rag-chatbot
   ```

4. Access the application at: http://localhost:8501


## Configuration

1.  **Environment Variables (Optional):** Create a `.env` file in the root directory to store API keys for OpenAI, Google, or Anthropic. You can also enter these directly in the Streamlit UI.
    ```
    OPENAI_API_KEY=your_openai_api_key_here
    GOOGLE_API_KEY=your_google_api_key_here
    ANTHROPIC_API_KEY=your_anthropic_api_key_here
    OLLAMA_BASE_URL=http://localhost:11434
    ```

2.  **Ollama (for local models ):**
    -   Install Ollama from [ollama.ai](https://ollama.ai/ ).
    -   Pull a model (e.g., `llama2`): `ollama pull llama2`
    -   Ensure Ollama is running on `http://localhost:11434`.

## Running the Application

1.  **Start the Streamlit app:**
    ```bash
    streamlit run main.py
    ```

2.  Open your web browser and navigate to the URL displayed in the terminal (usually `http://localhost:8501` ).

## Usage

1.  **Select LLM Model**: Choose your preferred LLM from the sidebar (OpenAI, Google, Anthropic, or Local).
2.  **Enter API Key**: If using a cloud-based model, provide your API key.
3.  **Upload Documents**: Use the file uploader to select PDF, TXT, or DOCX files.
4.  **Ingest Documents**: Click the "Ingest" button to process and embed your documents into the vector store.
5.  **Ask Questions**: Type your questions in the chat input box. The chatbot will retrieve relevant information from your documents and generate a response.
6.  **Clear Data**: Use the "Clear Data" button to remove all ingested documents and clear the chat history.
7.  **Clear Chat**: Use the "Clear Chat" button to clear only the chat messages.

## Logging

Application logs are saved to `logs/rag_chatbot.log` and also displayed in the console. The `logs` directory will be automatically created if it does not exist.

## Troubleshooting

-   **Import Errors**: Ensure all dependencies listed in `requirements.txt` are installed.
-   **API Key Issues**: Double-check your API keys for correctness and sufficient credits.
-   **Ollama Connection**: Verify that Ollama is running and accessible at the configured base URL.
-   **Document Processing**: Large documents may take longer to process during ingestion. Be patient.
-   **FAISS Errors**: If you encounter errors related to FAISS, ensure the `embeddings/faiss_index` directory is correctly managed (the application handles creation and clearing).

Feel free to contribute or report issues on the GitHub repository.
