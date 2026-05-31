import logging
import sys
import pytest
from pathlib import Path
from src.utils.logger import get_logger

def test_logger_initialization():
    """Verify basic naming and default level assignment."""
    import time
    name = f"test_unique_basic_init_{int(time.time())}"
    
    logger = get_logger(name)
    
    assert logger.name == name
    # We verify the logger has been configured (level is set)
    assert logger.level != logging.NOTSET
    assert logger.propagate is False
def test_logger_custom_level():
    """Verify that custom logging levels are correctly applied."""
    logger = get_logger("test_unique_level_custom", level=logging.DEBUG)
    assert logger.level == logging.DEBUG

def test_stream_handler_setup():
    """Check if a StreamHandler is attached to stdout."""
    logger = get_logger("test_unique_stream_handler")

    handlers = logger.handlers
    assert len(handlers) >= 1
    assert any(isinstance(h, logging.StreamHandler) for h in handlers)

    # Verify stream is stdout
    stream_handler = next(h for h in handlers if isinstance(h, logging.StreamHandler))
    assert stream_handler.stream == sys.stdout

def test_file_handler_creation(tmp_path):
    """Verify FileHandler attachment and directory creation using a temporary path."""
    log_file = tmp_path / "logs" / "test.log"
    logger = get_logger("test_unique_file_handler", log_file=str(log_file))

    # Verify the directory was created and the file exists
    assert log_file.exists()

    # Reference logger.handlers and simplify the generator
    assert any(isinstance(h, logging.FileHandler) for h in logger.handlers)

    # Retrieve the handler to verify the path
    file_handler = next(h for h in logger.handlers if isinstance(h, logging.FileHandler))
    assert Path(file_handler.baseFilename).resolve() == log_file.resolve()

def test_idempotency_handlers():
    """Ensure multiple calls to the same logger name do not duplicate handlers."""
    name = "test_unique_idempotency_handlers"
    logger1 = get_logger(name)
    initial_handler_count = len(logger1.handlers)

    logger2 = get_logger(name)
    assert len(logger2.handlers) == initial_handler_count

def test_formatter_config():
    """Verify the log format string is correctly assigned to handlers."""
    logger = get_logger("test_unique_formatter_config")
    expected_fmt = "[%(asctime)s] %(levelname)s [%(name)s:%(lineno)d] %(message)s"

    for handler in logger.handlers:
        assert handler.formatter._fmt == expected_fmt