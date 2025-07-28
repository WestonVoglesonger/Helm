#!/usr/bin/env python
"""Main entry point for the Helm agentic stack.

This module provides both CLI and HTTP interfaces for interacting with the
orchestrator. The CLI is designed for development and testing, while the
HTTP interface can be used for production deployments.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

import click
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .config import get_settings
from .services.orchestrator import Orchestrator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="Helm Agentic Stack", version="1.0.0")

# Request/Response models
class ChatRequest(BaseModel):
    user_id: str
    message: str

class ChatResponse(BaseModel):
    response: str
    user_id: str

# Global orchestrator instance
orchestrator: Optional[Orchestrator] = None

@app.on_event("startup")
async def startup_event():
    """Initialize the orchestrator on startup."""
    global orchestrator
    orchestrator = Orchestrator()
    logger.info("Orchestrator initialized")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Handle a chat request."""
    if orchestrator is None:
        raise HTTPException(status_code=500, detail="Orchestrator not initialized")
    
    try:
        response = await orchestrator.run_turn(request.user_id, request.message)
        return ChatResponse(response=response, user_id=request.user_id)
    except Exception as e:
        logger.error("Error processing chat request: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "orchestrator_ready": orchestrator is not None}

# CLI Commands
@click.group()
def cli():
    """Helm Agentic Stack CLI."""
    pass

@cli.command()
@click.option("--user-id", default="default", help="User ID for the session")
@click.option("--scenario", type=click.Path(exists=True), help="Path to scenario YAML file")
def chat(user_id: str, scenario: Optional[str]):
    """Start an interactive chat session."""
    if scenario:
        # Run scenario file
        asyncio.run(run_scenario(scenario))
    else:
        # Interactive mode
        asyncio.run(interactive_chat(user_id))

@cli.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=8000, help="Port to bind to")
@click.option("--reload", is_flag=True, help="Enable auto-reload")
def serve(host: str, port: int, reload: bool):
    """Start the HTTP server."""
    uvicorn.run("src.main:app", host=host, port=port, reload=reload)

async def interactive_chat(user_id: str):
    """Run an interactive chat session."""
    orch = Orchestrator()
    print(f"Starting chat session for user: {user_id}")
    print("Type 'quit' to exit")
    
    while True:
        try:
            message = input("\nYou: ").strip()
            if message.lower() in ['quit', 'exit', 'q']:
                break
            if not message:
                continue
            
            response = await orch.run_turn(user_id, message)
            print(f"Assistant: {response}")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

async def run_scenario(scenario_path: str):
    """Run a scenario from a YAML file."""
    import yaml
    from pathlib import Path
    
    orch = Orchestrator()
    
    with open(scenario_path, 'r') as f:
        scenario = yaml.safe_load(f)
    
    for turn in scenario:
        user_id = turn.get("user_id", "default")
        message = turn.get("message", "")
        print(f"\n[user {user_id}] {message}")
        
        response = await orch.run_turn(user_id, message)
        print(f"[assistant] {response}")

if __name__ == "__main__":
    cli() 