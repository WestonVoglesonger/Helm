"""Multi‑armed bandit algorithms for snippet selection.

This module currently implements a simple epsilon‑greedy bandit.  For each
user/task, the bandit maintains estimates of expected reward for each arm
(snippet).  The state can be persisted to the database to survive restarts.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Dict, List


logger = logging.getLogger(__name__)


@dataclass
class BanditState:
    """Holds reward estimates and counts for each arm."""

    values: Dict[str, float] = field(default_factory=dict)
    counts: Dict[str, int] = field(default_factory=dict)


class EpsilonGreedyBandit:
    """Simple epsilon‑greedy bandit.

    Args:
        arms: list of arm identifiers (e.g. snippet keys)
        epsilon: exploration probability
    """

    def __init__(self, arms: List[str], epsilon: float = 0.1) -> None:
        self.arms = arms
        self.epsilon = epsilon
        self.state = BanditState(
            values={arm: 0.0 for arm in arms},
            counts={arm: 0 for arm in arms},
        )

    def choose(self) -> str:
        """Select an arm based on epsilon‑greedy strategy."""
        explore = random.random() < self.epsilon
        if explore:
            choice = random.choice(self.arms)
            logger.debug("Bandit exploring; chose %s", choice)
            return choice
        # Exploit: pick arm with highest value estimate
        max_value = max(self.state.values.values())
        candidates = [arm for arm, value in self.state.values.items() if value == max_value]
        choice = random.choice(candidates)
        logger.debug("Bandit exploiting; chose %s", choice)
        return choice

    def update(self, arm: str, reward: float) -> None:
        """Update estimates for the selected arm.

        Uses a running average: new_value = (n * old + reward) / (n + 1).
        """
        if arm not in self.arms:
            logger.warning("Unknown arm %s", arm)
            return
        n = self.state.counts[arm]
        old_value = self.state.values[arm]
        new_value = (n * old_value + reward) / (n + 1)
        self.state.counts[arm] = n + 1
        self.state.values[arm] = new_value
        logger.debug("Bandit updated %s: count=%d, value=%.4f", arm, self.state.counts[arm], self.state.values[arm])

    def to_dict(self) -> Dict[str, Dict[str, float]]:
        """Serialize state for persistence."""
        return {
            "values": self.state.values,
            "counts": self.state.counts,
        }

    @classmethod
    def from_dict(cls, arms: List[str], data: Dict[str, Dict[str, float]], epsilon: float = 0.1) -> "EpsilonGreedyBandit":
        bandit = cls(arms, epsilon)
        bandit.state.values.update(data.get("values", {}))
        bandit.state.counts.update(data.get("counts", {}))
        return bandit