"""Prompt compiler for assembling system prompts.

Given user preferences and selected snippets, this module uses a Jinja2
template to build the final system prompt.  The default template is loaded
from `prompt_templates/system_prompt.j2`, but can be overridden.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .loader import TemplateLoader

logger = logging.getLogger(__name__)


class PromptCompiler:
    """Compile a system prompt from templates and context."""

    def __init__(self, loader: Optional[TemplateLoader] = None, template_name: str = "system_prompt.j2") -> None:
        self.loader = loader or TemplateLoader()
        self.template_name = template_name

    def compile(
        self,
        prefs: Dict[str, Any],
        snippets: Dict[str, Any],
        bandit_choices: Optional[List[str]] = None,
    ) -> str:
        """Render the system prompt.

        Args:
            prefs: The user/task preferences dictionary.
            snippets: A dictionary of available snippet contents keyed by name.
            bandit_choices: Optional list of snippet keys selected by the bandit.

        Returns:
            The rendered system prompt string.
        """
        template = self.loader.get_template(self.template_name)
        # Determine which snippets to include.  If no bandit choices provided,
        # include all snippets by default.
        selected = bandit_choices or list(snippets.keys())
        selected_snippets = {k: snippets[k] for k in selected if k in snippets}
        context = {
            "prefs": prefs,
            "snippets": selected_snippets,
        }
        try:
            return template.render(context)
        except Exception as exc:
            logger.error("Failed to render system prompt: %s", exc)
            raise