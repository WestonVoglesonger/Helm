import asyncio

import pytest

from src.prompt.loader import TemplateLoader
from src.prompt.compiler import PromptCompiler


@pytest.mark.asyncio
async def test_compile_basic() -> None:
    loader = TemplateLoader(template_dir="prompt_templates")
    compiler = PromptCompiler(loader=loader)
    prefs = {"tone": "friendly"}
    snippets = {"test": "Say hi."}
    result = compiler.compile(prefs, snippets, bandit_choices=["test"])
    assert "friendly" in result
    assert "Say hi." in result