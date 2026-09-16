"""Part 2, Task 3 — at least three Matplotlib figures, each saved to disk.

Run:  python -m capstone_part2.src.visualizations

Suggested set (pick >=3, each with a written interpretation in the report):
  1. Mean traffic volume by hour of day, weekday vs weekend overlaid.
  2. Mean traffic volume by weather_main, sorted, with n per category
     annotated -- Squall has 4 observations and that must be visible.
  3. Temperature vs traffic volume scatter (drop the 0 K rows first).
  4. Monthly record count over time -- this makes the 2014-2015 coverage
     gap visible and supports the Part 1 yearly-trend argument.

Logging: INFO with the file path each time a figure is saved.
"""

from __future__ import annotations

from common.logging_config import get_logger

logger = get_logger(__name__)

# TODO: one function per figure, each returning the saved path.
