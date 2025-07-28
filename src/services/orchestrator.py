"""Orchestration loop for the agentic stack.

The orchestrator coordinates retrieving preferences, compiling prompts,
executing LLM calls, running evaluations, and updating policies.  It is
designed to be called from a CLI or HTTP endpoint.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from ..config import get_settings
from ..llm.openai import OpenAIAdapter
from ..store.pref_store import PrefStore
from ..store.feature_store import FeatureStore
from ..prompt.loader import TemplateLoader
from ..prompt.compiler import PromptCompiler
from ..prompt.linter import lint_prompt
from ..policy.bandit import EpsilonGreedyBandit
from ..policy.updater import PolicyUpdater
from ..eval.heuristics import evaluate_response
from ..eval.eval_llm import eval_llm


logger = logging.getLogger(__name__)


class Orchestrator:
    """Coordinates the components for a single turn of interaction."""

    def __init__(self) -> None:
        settings = get_settings()
        self.pref_store = PrefStore()
        self.feature_store = FeatureStore()
        # Load snippets (placeholder YAML file) once
        self.loader = TemplateLoader()
        # For demonstration, load all snippet definitions from a YAML file
        try:
            self.snippets = self.loader.load_snippets("snippets.yaml")
        except FileNotFoundError:
            logger.warning(
                "snippets.yaml not found in prompt_templates; using empty snippets"
            )
            self.snippets = {}
        # Initialize bandit with snippet keys
        self.bandit = EpsilonGreedyBandit(list(self.snippets.keys()))
        self.policy_updater = PolicyUpdater(
            pref_store=self.pref_store,
            feature_store=self.feature_store,
            bandit=self.bandit,
        )
        self.compiler = PromptCompiler(loader=self.loader)
        self.llm = OpenAIAdapter()

    async def run_turn(self, user_id: str, user_msg: str) -> str:
        """Execute a single turn.

        Returns the model's response.
        """
        # Get preferences
        prefs = await self.pref_store.get(user_id)
        # Select snippet
        arm = None
        if self.snippets:
            arm = self.bandit.choose()
            selected = [arm]
        else:
            selected = None
        # Compile system prompt
        system_prompt = self.compiler.compile(
            prefs=prefs, snippets=self.snippets, bandit_choices=selected
        )
        # Lint prompt
        ok, reason = lint_prompt(system_prompt)
        if not ok:
            logger.error("Prompt failed linter: %s", reason)
            raise ValueError(f"Prompt failed linter: {reason}")
        # Call LLM
        model_response = await self.llm.call(
            system_prompt=system_prompt, user_msg=user_msg, tools=None
        )
        # Evaluate response
        reward, metrics = evaluate_response(system_prompt, user_msg, model_response)
        # Combine heuristic reward with optional eval LLM reward
        eval_reward = await eval_llm(system_prompt, user_msg, model_response)
        combined_reward = 0.5 * reward + 0.5 * eval_reward
        # Update policy
        await self.policy_updater.update(
            user_id=user_id,
            reward=combined_reward,
            arm=arm,
            features=metrics,
        )
        return model_response