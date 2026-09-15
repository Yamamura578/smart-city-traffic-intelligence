"""Central logging configuration for the capstone project.

Design rules (these map directly onto the capstone logging requirements):

* Library/analysis modules NEVER configure handlers. They only call
  ``get_logger(__name__)`` and log. This keeps them importable from
  notebooks, tests or other scripts without side effects.
* Handlers are attached exactly once, by the entry-point script, via
  ``setup_logging()``.
* Every run writes to BOTH the console and a log file, using a formatter
  that carries timestamp, level, module name and message.
* DEBUG output is opt-in: pass ``debug=True`` (wired to a ``--debug`` CLI
  flag) so that intermediate values stay out of normal runs.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# timestamp | level | module | message  -- required minimum fields
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(log_file: Path | str, debug: bool = False) -> logging.Logger:
    """Configure the root logger with a console and a file handler.

    Call this ONCE, from an entry-point script (``if __name__ == "__main__"``).
    Never call it from an imported analysis module.

    Parameters
    ----------
    log_file:
        Destination for the file handler, e.g. ``logs/pipeline.log``.
        Parent directories are created if missing.
    debug:
        When True the threshold drops to DEBUG so intermediate values
        (quartile thresholds, imputation values, ...) become visible.
        When False only INFO and above are emitted.

    Returns
    -------
    The configured root logger (rarely needed; modules should use
    ``get_logger(__name__)`` instead).
    """
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    level = logging.DEBUG if debug else logging.INFO
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    root = logging.getLogger()
    root.setLevel(level)

    # Re-running in the same interpreter (e.g. a notebook) would otherwise
    # stack duplicate handlers and print every message twice.
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    console = logging.StreamHandler(stream=sys.stdout)
    console.setLevel(level)
    console.setFormatter(formatter)
    root.addHandler(console)

    # mode="a" keeps a cumulative audit trail across runs, which is what the
    # graders are asked to review. Switch to "w" if you prefer one run per file.
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    root.info("Logging configured | level=%s | file=%s", logging.getLevelName(level), log_file)
    return root


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger.

    Always call as ``get_logger(__name__)`` so the module name appears in
    the log line and log output can be traced back to its source file.
    """
    return logging.getLogger(name)
