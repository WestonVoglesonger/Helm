import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from src.services.orchestrator import Orchestrator


@pytest.mark.asyncio
async def test_orchestrator_run_turn(monkeypatch) -> None:
    orch = Orchestrator()
    # Patch llm.call to avoid calling external API
    async def fake_call(system_prompt: str, user_msg: str, tools=None) -> str:
        return "fake response"
    monkeypatch.setattr(orch.llm, "call", fake_call)
    # Patch eval_llm to avoid OpenAI call
    monkeypatch.setattr("src.eval.eval_llm.eval_llm", AsyncMock(return_value=0.0))
    # Run
    response = await orch.run_turn("u1", "Hello")
    assert response == "fake response"