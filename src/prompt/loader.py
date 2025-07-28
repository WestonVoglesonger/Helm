"""Loader for prompt templates and snippet definitions.

Templates live under the `prompt_templates/` directory.  This module uses
Jinja2's `Environment` to load and render templates.  Snippets are defined
in YAML files alongside the templates.
"""

from __future__ import annotations

import logging
import os
from importlib import resources
from pathlib import Path
from typing import Any, Dict

import jinja2
import yaml


logger = logging.getLogger(__name__)


class TemplateLoader:
    """Loads Jinja2 templates and YAML snippets from disk."""

    def __init__(self, template_dir: str = "prompt_templates") -> None:
        # Determine absolute path
        base_path = Path(os.getcwd()) / template_dir
        self.template_dir = base_path
        self.env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(base_path)), autoescape=False
        )

    def get_template(self, name: str) -> jinja2.Template:
        """Return a Jinja2 template by name."""
        try:
            return self.env.get_template(name)
        except Exception as exc:
            logger.error("Failed to load template %s: %s", name, exc)
            raise

    def load_snippets(self, yaml_name: str) -> Dict[str, Any]:
        """Load a YAML file containing snippet definitions."""
        path = self.template_dir / yaml_name
        if not path.exists():
            raise FileNotFoundError(f"Snippet YAML not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)