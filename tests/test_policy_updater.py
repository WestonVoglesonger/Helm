import asyncio
import pytest

from src.store.pref_store import PrefStore
from src.store.feature_store import FeatureStore
from src.policy.bandit import EpsilonGreedyBandit
from src.policy.updater import PolicyUpdater


@pytest.mark.asyncio
async def test_policy_updater_updates_bandit(tmp_path) -> None:
    # Create isolated feature store in temp directory
    feature_path = tmp_path / "features.parquet"
    feature_store = FeatureStore(path=str(feature_path))
    pref_store = PrefStore()
    bandit = EpsilonGreedyBandit(["arm1", "arm2"], epsilon=0.0)
    updater = PolicyUpdater(pref_store, feature_store, bandit)
    await updater.update(
        user_id="u1",
        reward=0.8,
        arm="arm1",
        features={"len_tokens": 10, "pref_violation_count": 0, "latency_ms": 5.0},
    )
    assert bandit.state.counts["arm1"] == 1
    assert bandit.state.values["arm1"] == pytest.approx(0.8)