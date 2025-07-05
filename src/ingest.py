import os
import json
from typing import List
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, UnstructuredPowerPointLoader
from langchain.schema import Document
from langdetect import detect
from .config import Config
import logging
import tiktoken


def get_embeddings():
    """Initialize embeddings with proper device configuration"""
    # Set environment variables to force CPU usage
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

    return HuggingFaceEmbeddings(
        model_name=Config.EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )


def load_document(file_path: str, logger: logging.Logger):
    logger.info(f"Attempting to load file: {file_path}")

    try:
        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            logger.debug("Using PyPDFLoader")
        elif file_path.endswith(".txt") or file_path.endswith(".md"):
            loader = TextLoader(file_path, encoding="utf-8")
            logger.debug("Using TextLoader for .txt or .md")
        elif file_path.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
            logger.debug("Using Docx2txtLoader")
        elif file_path.endswith(".ppt") or file_path.endswith(".pptx"):
            loader = UnstructuredPowerPointLoader(file_path)
            logger.debug("Using UnstructuredPowerPointLoader")
        else:
            logger.warning(f"Unsupported file type: {file_path}")
            raise ValueError(f"Unsupported file type: {file_path}")

        documents = loader.load()
        logger.info(f"Successfully loaded {len(documents)} document(s) from {file_path}")
        return documents

    except Exception as e:
        logger.error(f"Error loading document {file_path}: {str(e)}")
        raise


def process_documents(documents: List[Document], original_filename: str, logger: logging.Logger):
    """Process documents and split into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )

    processed_chunks = []
    for doc in documents:
        chunks = text_splitter.split_documents([doc])
        for chunk in chunks:
            chunk.metadata['source'] = original_filename
            processed_chunks.append(chunk)

    logger.info(f"Created {len(processed_chunks)} chunks from {original_filename}")
    return processed_chunks


def count_words(text: str) -> int:
    enc = tiktoken.get_encoding("cl100k_base")
    return len(text.split())


def ingest_documents(file_paths: List[str], original_filenames: List[str], logger: logging.Logger):
    all_processed_chunks = []
    documents_metadata = []

    for file_path, original_filename in zip(file_paths, original_filenames):
        logger.info(f"Loading document: {original_filename}")
        try:
            documents = load_document(file_path, logger)
            processed_chunks = process_documents(documents, original_filename, logger)
            all_processed_chunks.extend(processed_chunks)

            # Store metadata for each document
            documents_metadata.append({
                "file_name": original_filename,
                "word_count": sum(count_words(doc.page_content) for doc in documents),
                "language": detect(documents[0].page_content) if documents else "unknown"
            })
        except Exception as e:
            logger.error(f"Failed to load or process document {original_filename}: {e}")
            raise

    if all_processed_chunks:
        # Initialize embeddings only when needed
        logger.info("Initializing embeddings...")
        embeddings = get_embeddings()
        logger.info("Embeddings initialized successfully")

        # Create or load existing vector store
        if os.path.exists(Config.FAISS_INDEX_PATH):
            logger.info("Loading existing FAISS index")
            vectorstore = FAISS.load_local(Config.FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
            vectorstore.add_documents(all_processed_chunks)
            logger.info(f"Added {len(all_processed_chunks)} chunks to existing vector store")
        else:
            logger.info("Creating new FAISS index")
            vectorstore = FAISS.from_documents(all_processed_chunks, embeddings)
            logger.info(f"Created new vector store with {len(all_processed_chunks)} chunks")

        # Save the updated vector store
        vectorstore.save_local(Config.FAISS_INDEX_PATH)
        logger.info("Vector store saved successfully")

        # Update documents metadata
        existing_metadata = []
        if os.path.exists(Config.DOCUMENTS_JSON_PATH):
            with open(Config.DOCUMENTS_JSON_PATH, 'r') as f:
                existing_metadata = json.load(f)

        existing_metadata.extend(documents_metadata)

        with open(Config.DOCUMENTS_JSON_PATH, 'w') as f:
            json.dump(existing_metadata, f, indent=2)

        logger.info(f"Updated metadata for {len(documents_metadata)} documents")

    return all_processed_chunks


def clear_ingested_data(logger: logging.Logger):
    """Clear all ingested data"""
    try:
        if os.path.exists(Config.FAISS_INDEX_PATH):
            import shutil
            shutil.rmtree(Config.FAISS_INDEX_PATH)
            logger.info("Removed FAISS index directory")

        if os.path.exists(Config.DOCUMENTS_JSON_PATH):
            os.remove(Config.DOCUMENTS_JSON_PATH)
            logger.info("Removed documents metadata file")

        logger.info("All ingested data cleared successfully")
    except Exception as e:
        logger.error(f"Error clearing ingested data: {e}")
        raise