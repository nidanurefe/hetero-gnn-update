from __future__ import annotations

import logging
from pathlib import Path
from datetime import datetime

# Global state to ensure single initialization
_log_file = None
_initialized = False


def setup_logging(
    log_dir: str | Path = "logs",
    level: int = logging.INFO,
    force_new: bool = False,
) -> logging.Logger:

    global _log_file, _initialized
    
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create log file only once 
    if _log_file is None or force_new:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        _log_file = log_dir / f"run_{ts}.log"

    # Get root logger
    logger = logging.getLogger()
    logger.setLevel(level)

    # Only add handlers once
    if not _initialized or force_new:
        # Clear existing handlers
        logger.handlers.clear()
        
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level)

        # File handler
        file_handler = logging.FileHandler(_log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
        _initialized = True
        logger.info(f"Logging initialized. Log file: {_log_file}")

    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    if not _initialized:
        setup_logging()
    return logging.getLogger(name)