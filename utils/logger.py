"""
MSNet — Logging Setup
=======================
Configures the Python logging system with:
  - Color-coded console output (INFO=green, WARNING=yellow, ERROR=red)
  - File logging with rotation
  - Timestamp + module name in each message
  - Compatible with W&B and TensorBoard
"""

import logging
import sys
from pathlib import Path


class ColorFormatter(logging.Formatter):
    """Custom formatter with ANSI color codes for console output."""

    COLORS = {
        logging.DEBUG:   "\033[36m",   # Cyan
        logging.INFO:    "\033[32m",   # Green
        logging.WARNING: "\033[33m",   # Yellow
        logging.ERROR:   "\033[31m",   # Red
        logging.CRITICAL:"\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelno, self.RESET)
        formatted = super().format(record)
        return f"{color}{formatted}{self.RESET}"


def setup_logging(
    log_level: str = "INFO",
    log_file: str = None,
) -> logging.Logger:
    """
    Configure root logger with console + optional file handler.

    Args:
        log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        log_file: Optional file path for persistent log file

    Returns:
        Root logger
    """
    level = getattr(logging, log_level.upper(), logging.INFO)

    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(
        ColorFormatter(
            fmt="%(asctime)s [%(levelname)-8s] %(name)-25s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )

    handlers = [console_handler]

    # File handler
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s [%(levelname)-8s] %(name)s — %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        handlers.append(file_handler)

    logging.basicConfig(
        level=level,
        handlers=handlers,
        force=True,
    )

    # Suppress noisy third-party loggers
    for noisy_lib in ["PIL", "matplotlib", "urllib3", "wandb"]:
        logging.getLogger(noisy_lib).setLevel(logging.WARNING)

    return logging.getLogger()
