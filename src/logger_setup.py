import logging
import os

def setup_logging(log_level=logging.INFO):
    """
    Sets up logging for chatbot application.
    Logs are then saved to logs/rag_chatbot.log and also printed in the console.
    The log directory will be created if it doesn't exist.

    Args:
        log_level: The minimum logging level to capture (e.g., logging.INFO, logging.DEBUG).
    """
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, "rag_chatbot.log")

    # Creates a logger
    logger = logging.getLogger("rag_chatbot_logger")
    logger.setLevel(log_level)

    # Creates a file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)

    # Creates a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)

    # Creates a formatter and adds it to the handlers
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Adds the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger