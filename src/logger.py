"""Logging configuration for MedGemma AI"""

import logging
import os
import sys

class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels"""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        # Add color to level name and pad for alignment
        if record.levelname in self.COLORS:
            # Pad level name to 8 characters for alignment (WARNING is longest at 7)
            padded_level = record.levelname.ljust(8)
            record.levelname = f"{self.COLORS[record.levelname]}{padded_level}{self.RESET}"
        return super().format(record)

def setup_logger(name: str) -> logging.Logger:
    """Setup logger with consistent formatting and colors"""
    # Allow DEBUG mode via environment variable
    log_level = logging.DEBUG if os.getenv("DEBUG", "").lower() in ("1", "true", "yes") else logging.INFO
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    
    # Use colored formatter for console
    formatter = ColoredFormatter(
        fmt='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger