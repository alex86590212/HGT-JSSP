"""Compatibility wrapper for the modular FCFS TraCI controller."""

from __future__ import annotations

from src.sumo_interface.traci_runner import *  # noqa: F401,F403
from src.sumo_interface.traci_runner import parse_args, run_controller


if __name__ == "__main__":
    print("Compatibility note: use `python -m src.main --mode headless --scheduler fcfs` for new runs.")
    run_controller(parse_args())
