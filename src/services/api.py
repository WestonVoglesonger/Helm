"""FastAPI application exposing the orchestration loop.

This module defines an optional HTTP layer around the orchestrator.  It allows
clients to send messages via REST.  To run the API, use:

```bash
uvicorn src.services.api:app --reload
```
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .orchestrator import Orchestrator


logger = logging.getLogger(__name__)

app = FastAPI(title="PipelineBuilder API")
orchestrator = Orchestrator()


class ChatRequest(BaseModel):
    user_id: str
    message: str


class ChatResponse(BaseModel):
    response: str


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest) -> ChatResponse:
    try:
        result = await orchestrator.run_turn(req.user_id, req.message)
        return ChatResponse(response=result)
    except Exception as exc:
        logger.exception("Error in chat endpoint: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))