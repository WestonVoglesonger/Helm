from src.policy.bandit import EpsilonGreedyBandit


def test_epsilon_greedy_bandit() -> None:
    arms = ["a", "b"]
    bandit = EpsilonGreedyBandit(arms, epsilon=0.0)
    # Without any updates, values are equal; choose() should return one of them
    choice = bandit.choose()
    assert choice in arms
    # Update one arm
    bandit.update("a", 1.0)
    # Now the highest value is for arm 'a'; choose should return 'a'
    for _ in range(10):
        assert bandit.choose() == "a"