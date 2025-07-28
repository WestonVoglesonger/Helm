#!/usr/bin/env python
"""Demo runner script.

This script reads a YAML scenario describing a sequence of interactions and
replays them through the orchestrator.  It prints the model's responses to
stdout.  Use this to exercise the full loop during development and CI.

Example scenario YAML:

```yaml
- user_id: alice
  message: "Hello, assistant!"
- user_id: alice
  message: "What can you do for me?"
```
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path
from typing import List, Dict

import yaml

from src.services.orchestrator import Orchestrator


logger = logging.getLogger(__name__)


def load_scenario(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, list):
        raise ValueError("Scenario YAML must contain a list of turns")
    return data


async def run_scenario(orchestrator: Orchestrator, scenario: List[Dict[str, str]]) -> None:
    for turn in scenario:
        user_id = turn.get("user_id", "default")
        message = turn.get("message", "")
        print(f"\n[user {user_id}] {message}")
        response = await orchestrator.run_turn(user_id, message)
        print(f"[assistant] {response}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a demo scenario")
    parser.add_argument(
        "--scenario",
        type=Path,
        required=True,
        help="Path to YAML file containing the scenario",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, etc.)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level.upper())
    orchestrator = Orchestrator()
    scenario = load_scenario(args.scenario)
    asyncio.run(run_scenario(orchestrator, scenario))


if __name__ == "__main__":
    main()