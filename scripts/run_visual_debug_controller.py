"""Compatibility wrapper for the modular live graph debugger."""

from __future__ import annotations

from src.visualization.graph_debugger import *  # noqa: F401,F403
from src.visualization.graph_debugger import parse_args, run_controller_with_visualization


if __name__ == "__main__":
    print("Compatibility note: use `python -m src.main --mode debug --scheduler fcfs` for new runs.")
    run_controller_with_visualization(parse_args())
