from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_models import ChatOllama
from langchain_anthropic import ChatAnthropic
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import logging

from .config import Config
from .retriever import get_vectorstore

# Code to select the desired LLM and its API Key you can add more models if you need
def get_llm(model_name: str, api_key: str = None):
    if model_name == "OpenAI":
        return ChatOpenAI(model="gpt-3.5-turbo", api_key=api_key)
    elif model_name == "Google":
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=api_key)
    elif model_name == "Anthropic":
        return ChatAnthropic(model="claude-2", api_key=api_key)
    elif model_name == "Local":
        return ChatOllama(model="llama3", base_url=Config.OLLAMA_BASE_URL)
    else:
        raise ValueError("Unsupported LLM model")

# Code to generate a response from the selected model, include the System Prompt
def generate_response(model_name: str, query: str, api_key: str = None, logger: logging.Logger = None):
    if logger is None:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())

    logger.info(f"Generating response for query: {query} using model: {model_name}")
    llm = get_llm(model_name, api_key)
    vectorstore = get_vectorstore()

    if not vectorstore:
        logger.warning("No documents ingested yet, cannot generate response.")
        return "No documents have been ingested yet. Please upload and ingest documents first."

    retriever = vectorstore.as_retriever()
    logger.info("Retriever initialized.")
    
    # The main System Prompt is defined here, you can change it you you want
    system_prompt = (
        "You are an AI assistant for question-answering tasks. "
        "Use the following retrieved context to answer the question. "
        "If you do not know the answer, just say that you do not know. "
        "Use three sentences maximum and keep the answer concise.\n\n"
        "{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    qna_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, qna_chain)

    try:
        response = rag_chain.invoke({"input": query})
        logger.info("RAG chain invoked successfully.")
        return response["answer"]
    except Exception as e:
        logger.error(f"Error during RAG chain invocation: {e}")
        return f"An error occurred while generating the response: {str(e)}"
