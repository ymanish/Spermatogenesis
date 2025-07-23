import logging
import sys

def get_logger(name: str, log_file: str = None, level: int = logging.INFO) -> logging.Logger:
    """
    Create and return a logger with the specified name.
    
    Parameters:
        name (str): Name for the logger.
        log_file (str): If provided, log messages will be written to this file.
        level (int): Logging level (e.g., logging.INFO, logging.DEBUG).
        
    Returns:
        logging.Logger: A configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Clear any existing handlers
    logger.handlers.clear()
    
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    # Create a stream handler to output log messages to the terminal
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # If a log file path is provided, add a file handler
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    return logger

if __name__ == "__main__":
    logger = get_logger("MyLogger", log_file="example.log", level=logging.DEBUG)
    logger.debug("This is a debug message.")
    logger.info("This is an informational message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")
    logger.critical("This is a critical message.")