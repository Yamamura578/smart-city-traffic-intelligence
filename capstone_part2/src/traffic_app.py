"""Part 2, Task 4 — command-line traffic analytics application.

Run:  python -m capstone_part2.src.traffic_app <command> [args]

Needs at least three commands. Suggested:
  lookup   --datetime "2017-05-03 08:00"   -> volume + weather for that hour
  peak     --day-type weekday              -> busiest and quietest hours
  compare  --weather Clear --weather Snow  -> mean volume by condition
  advise   --day-type weekday              -> recommended travel window
                                              (this one feeds Part 3, Task 5)

Logging rules for this file specifically:
  * INFO: which command was invoked and with which arguments.
  * ERROR: a clear message for bad user input (e.g. an unparseable date) --
    not a raw traceback.
  * print() IS allowed here, but only for the answer the user asked for.
    Never for internal progress.
"""

from __future__ import annotations

import argparse

from common.logging_config import get_logger

logger = get_logger(__name__)

# TODO: argparse subparsers, one handler function per command.
