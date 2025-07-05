import streamlit as st
import os
import json
import tempfile
import logging
from .generator import generate_response
from .ingest import ingest_documents, clear_ingested_data
from .config import Config
from .logger_setup import setup_logging


# Initialize logger at module level
logger = setup_logging(logging.INFO)


# Clears the entire chat history
def clear_chat_history():
    logger.info("Clearing chat history")
    if "messages" in st.session_state:
        message_count = len(st.session_state.messages)
        st.session_state.messages = []
        logger.info(f"Cleared {message_count} messages from chat history")
    else:
        logger.info("No chat history to clear")


# Clears all the uploaded documents/files
def clear_uploaded_files():
    """Clear uploaded files from session state"""
    logger.info("Clearing uploaded files from session state")
    if "uploaded_files" in st.session_state:
        del st.session_state.uploaded_files
        logger.info("Uploaded files cleared from session state")
    else:
        logger.info("No uploaded files to clear from session state")


def main():
    logger.info("Starting Multilingual RAG Chatbot interface")

    try:
        catrag_ico = os.path.join("static", "CatRag.ico")  # Path for the catrag ico file ;)
        st.set_page_config(
            page_title="Multilingual RAG Chatbot",
            page_icon=catrag_ico,
            layout="wide"
        )

        st.title("Multilingual RAG Chatbot")
        st.markdown("Ask questions about your uploaded documents in any language!")

        # Code in Sidebar here
        with st.sidebar:
            st.header("Configuration")

            with st.expander("Model Setting"):
                # Model Selector
                model_options = ["OpenAI", "Google", "Anthropic", "Local"]
                selected_model = st.selectbox("Select LLM Model", model_options)
                logger.debug(f"Selected model: {selected_model}")

                api_key = None
                if selected_model != "Local":
                    api_key = st.text_input(f"{selected_model} API Key", type="password")
                    if api_key:
                        logger.info(f"API key provided for {selected_model}")
                    else:
                        logger.warning(f"No API key provided for {selected_model}")

            st.divider()

            st.header("Upload Document")
            uploaded_files = st.file_uploader(
                "Choose File/s",
                type=["pdf", "txt", "docx"],
                accept_multiple_files=True,
                help="Upload PDF or TXT OR DOC files to chat with",
                key="file_uploader"
            )

            if uploaded_files:
                logger.info(f"User uploaded {len(uploaded_files)} files: {[f.name for f in uploaded_files]}")

            col1, col2 = st.columns(2)
            with col1:
                # Code for the Ingest Button
                if st.button("Ingest", type="primary"):
                    logger.info("Ingest button clicked")
                    if uploaded_files:
                        try:
                            # Loads previously ingested filenames
                            ingested_filenames = set()
                            if os.path.exists(Config.DOCUMENTS_JSON_PATH):
                                with open(Config.DOCUMENTS_JSON_PATH, "r") as f:
                                    ingested_docs = json.load(f)
                                    ingested_filenames = {doc["file_name"] for doc in ingested_docs}
                                logger.info(f"Found {len(ingested_filenames)} previously ingested files")

                            # Filters out already ingested files so that the existing ones do not ingest again
                            new_uploaded_files = [f for f in uploaded_files if f.name not in ingested_filenames]
                            logger.info(f"Found {len(new_uploaded_files)} new files to ingest")

                            if not new_uploaded_files:
                                st.info("All uploaded files are already ingested.")
                                logger.info("All uploaded files are already ingested")
                            else:
                                with st.spinner("Processing new documents..."):
                                    temp_paths = []
                                    og_names = []
                                    for uploaded_file in new_uploaded_files:
                                        with tempfile.NamedTemporaryFile(delete=False,
                                                                         suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                                            tmp_file.write(uploaded_file.getvalue())
                                            temp_paths.append(tmp_file.name)
                                            og_names.append(uploaded_file.name)

                                    logger.info(f"Created temporary files for {len(temp_paths)} documents")

                                    try:
                                        ingest_documents(temp_paths, og_names, logger)  # <-- pass both lists
                                        st.success(f"Successfully ingested {len(new_uploaded_files)} new document(s)!")
                                        logger.info(f"Successfully ingested {len(new_uploaded_files)} new document(s)")
                                    except Exception as e:
                                        error_msg = f"Error ingesting documents: {str(e)}"
                                        st.error(error_msg)
                                        logger.error(error_msg, exc_info=True)
                                    finally:
                                        for temp_path in temp_paths:
                                            try:
                                                os.unlink(temp_path)
                                                logger.debug(f"Deleted temporary file: {temp_path}")
                                            except Exception as e:
                                                logger.warning(f"Failed to delete temporary file {temp_path}: {e}")

                        except Exception as e:
                            error_msg = f"Unexpected error during ingestion process: {str(e)}"
                            st.error(error_msg)
                            logger.error(error_msg, exc_info=True)
                    else:
                        st.warning("Please upload documents first.")
                        logger.warning("User tried to ingest without uploading documents")

            with col2:
                if st.button("Clear Data"):
                    logger.info("Clear Data button clicked")
                    try:
                        clear_ingested_data(logger)
                        logger.info("Ingested data cleared")
                        clear_uploaded_files()
                        clear_chat_history()

                        st.success("All data and uploaded files cleared!")
                        logger.info("All data and uploaded files cleared successfully")
                        st.rerun()
                    except Exception as e:
                        error_msg = f"Error clearing data: {str(e)}"
                        st.error(error_msg)
                        logger.error(error_msg, exc_info=True)

            st.divider()

            st.header("Ingested Documents")
            try:
                if os.path.exists(Config.DOCUMENTS_JSON_PATH):
                    with open(Config.DOCUMENTS_JSON_PATH, 'r') as f:
                        docs_metadata = json.load(f)

                    if docs_metadata:
                        logger.debug(f"Displaying {len(docs_metadata)} ingested documents")
                        for doc in docs_metadata:
                            st.write(f"   **{doc['file_name']}**")
                            st.write(f"   - Words: {doc['word_count']}")
                            st.write(f"   - Language: {doc['language']}")
                    else:
                        st.write("No documents ingested yet.")
                        logger.debug("No documents in metadata file")
                else:
                    st.write("No documents ingested yet.")
                    logger.debug("Documents metadata file does not exist")
            except Exception as e:
                error_msg = f"Error reading ingested documents: {str(e)}"
                st.error(error_msg)
                logger.error(error_msg, exc_info=True)

        # COde for the Main Chat interface
        if "messages" not in st.session_state:
            st.session_state.messages = []
            logger.info("Initialized empty chat messages in session state")

        col1, col2 = st.columns([6, 1])
        with col2:
            if st.button("Clear Chat", help="Clear all chat messages"):
                logger.info("Clear Chat button clicked")
                clear_chat_history()
                st.success("Chat history cleared!")
                st.rerun()

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask a question about your document/s...."):
            logger.info(f"User submitted prompt: {prompt[:100]}..." if len(
                prompt) > 100 else f"User submitted prompt: {prompt}")

            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    if selected_model != "Local" and not api_key:
                        response = f"Please provide an API key for {selected_model}."
                        logger.warning(f"No API key provided for {selected_model} model")
                    else:
                        try:
                            logger.info(f"Generating response using {selected_model} model")
                            response = generate_response(selected_model, prompt, api_key)
                            logger.info(f"Successfully generated response (length: {len(response)} characters)")
                        except Exception as e:
                            response = f"Error generating response: {str(e)}"
                            logger.error(f"Error generating response: {str(e)}", exc_info=True)

                    st.markdown(response)

            st.session_state.messages.append({"role": "assistant", "content": response})
            logger.debug("Added assistant response to chat history")

    except Exception as e:
        error_msg = f"Critical error in main interface: {str(e)}"
        st.error(error_msg)
        logger.critical(error_msg, exc_info=True)


if __name__ == "__main__":
    main()