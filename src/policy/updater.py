"""Policy updater.

This component updates user preferences and bandit state based on evaluation
signals.  It receives a reward score and other metrics, persists them via
FeatureStore, and updates the bandit accordingly.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from ..store.pref_store import PrefStore
from ..store.feature_store import FeatureStore
from .bandit import EpsilonGreedyBandit


logger = logging.getLogger(__name__)


class PolicyUpdater:
    def __init__(self, pref_store: PrefStore, feature_store: FeatureStore, bandit: EpsilonGreedyBandit) -> None:
        self.pref_store = pref_store
        self.feature_store = feature_store
        self.bandit = bandit

    async def update(
        self,
        user_id: str,
        reward: float,
        arm: Optional[str] = None,
        features: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update the system based on reward and optional features.

        Args:
            user_id: Identifier for the user or task.
            reward: Primary reward signal from evaluation.
            arm: The bandit arm (snippet) used in this turn; if provided,
                update the bandit's estimate.
            features: Additional metrics to log (see evaluation signals schema).
        """
        # Persist reward in feature store
        record = {
            "user_id": user_id,
            "reward_primary": reward,
        }
        if features:
            record.update(features)
        self.feature_store.append(record)
        # Update bandit state
        if arm is not None:
            self.bandit.update(arm, reward)
        # TODO: update preferences in PrefStore if required
        # For example, increment a success counter or update embeddings
        logger.debug("Policy updated for user %s with reward %.4f", user_id, reward)