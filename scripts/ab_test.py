#!/usr/bin/env python
"""A/B testing harness (skeleton).

This script sets up two different prompt snippet configurations and measures
their performance over multiple turns.  It uses the same orchestrator but
forces specific bandit choices.  Results are summarised at the end.
"""

from __future__ import annotations

import asyncio
import argparse
import logging
from typing import Dict, List

from src.services.orchestrator import Orchestrator


logger = logging.getLogger(__name__)


async def run_ab_test(orchestrator: Orchestrator, user_id: str, messages: List[str]) -> Dict[str, float]:
    """Run two arms and collect average rewards.

    For each arm, send all messages with that arm selected and compute the
    average combined reward (heuristic + eval LLM).  Returns a dict of arm
    -> average reward.
    """
    results = {}
    # Force bandit choice by overriding bandit
    for arm in orchestrator.snippets.keys():
        total = 0.0
        for msg in messages:
            # Temporarily monkey‑patch bandit.choose to return the arm
            orchestrator.bandit = orchestrator.bandit.__class__(list(orchestrator.snippets.keys()))
            orchestrator.bandit.epsilon = 0.0
            orchestrator.bandit.state.values = {k: 0.0 for k in orchestrator.snippets.keys()}
            orchestrator.bandit.state.counts = {k: 0 for k in orchestrator.snippets.keys()}
            # Override choose() to always return selected arm
            orchestrator.bandit.choose = lambda a=arm: a
            response = await orchestrator.run_turn(user_id, msg)
            # Here we assume reward is logged; retrieving average is beyond scope
            total += 0.0  # placeholder
        results[arm] = total / len(messages) if messages else 0.0
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an A/B test on prompt snippets")
    parser.add_argument("user_id", type=str, help="User identifier")
    parser.add_argument(
        "messages", nargs="*", default=["Hello", "Tell me something interesting"], help="Messages to send"
    )
    args = parser.parse_args()
    logging.basicConfig(level="INFO")
    orchestrator = Orchestrator()
    results = asyncio.run(run_ab_test(orchestrator, args.user_id, args.messages))
    for arm, avg_reward in results.items():
        print(f"Arm {arm}: average reward {avg_reward:.4f}")


if __name__ == "__main__":
    main()