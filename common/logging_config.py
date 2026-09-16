"""Central logging configuration for the capstone project."""
from __future__ import annotations
import logging, sys
from pathlib import Path

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(log_file, debug: bool = False) -> logging.Logger:
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    level = logging.DEBUG if debug else logging.INFO
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler); handler.close()
    console = logging.StreamHandler(stream=sys.stdout)
    console.setLevel(level); console.setFormatter(formatter); root.addHandler(console)
    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setLevel(level); fh.setFormatter(formatter); root.addHandler(fh)
    # Third-party libraries are chatty at DEBUG (matplotlib's font manager
    # alone emits hundreds of lines per figure). Pin them to WARNING so that
    # --debug shows this project's own diagnostics and nothing else.
    for noisy in ("matplotlib", "PIL", "fontTools", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    root.info("Logging configured | level=%s | file=%s", logging.getLevelName(level), log_file)
    return root


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
