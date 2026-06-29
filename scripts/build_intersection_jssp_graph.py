"""Compatibility wrapper for the static JSSP graph example builder."""

from __future__ import annotations

from src.graph.static_jssp_example import *  # noqa: F401,F403
from src.graph.static_jssp_example import main


if __name__ == "__main__":
    print("Compatibility note: static graph code now lives in `src.graph.static_jssp_example`.")
    main()
