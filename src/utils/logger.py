import logging
import sys
from pathlib import Path

def get_logger(name: str, log_file: str = None, level: int = logging.INFO) -> logging.Logger:
    """
    Configures and returns a logger instance.
    
    Args:
        name: The name of the logger (usually __name__).
        log_file: Optional path to a file where logs will be saved.
        level: Logging level (e.g., logging.DEBUG, logging.INFO).
    """
    logger = logging.getLogger(name)
    
    # Prevents adding multiple handlers to the same logger if the 
    # function is called multiple times in the same session.
    if not logger.handlers:
        logger.setLevel(level)
        
        # Comprehensive format for debugging modular code
        formatter = logging.Formatter(
            fmt="[%(asctime)s] %(levelname)s [%(name)s:%(lineno)d] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # Console Output
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File Output (optional)
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    # Prevent logs from being propagated to the root logger
    logger.propagate = False
    return logger