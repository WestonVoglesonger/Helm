# PipelineBuilder Agent

This repository implements a two‑layer agentic stack built on top of a frozen base LLM.  The outer layer adapts prompts in real time based on user preferences and interaction history, while the inner layer calls a base language model (e.g. OpenAI) to generate responses.  The design emphasises modularity, testability and observability.

## Features

- **Unified configuration** via a Pydantic `Settings` object injected throughout the system.
- **Preference store** backed by PostgreSQL (JSONB) with a Redis cache; separate vector store using `pgvector`.
- **Prompt compiler** and linter using Jinja2 templates and token budget guards.
- **Bandit/RL‑lite** module with persisted state to select prompt snippets.
- **Evaluation module** emitting a structured reward schema.
- **Guardrails** applied both before sending prompts and after receiving outputs.
- **CLI and optional FastAPI** entrypoints.
- **Docker Compose** for local development (PostgreSQL + Redis).

## Quick Start

1. Copy `.env.example` to `.env` and fill in the required environment variables (database URLs, OpenAI API key, etc.).
2. Build and start the services using Docker Compose:

```bash
docker-compose up --build
```

3. Run the demo script, which will replay a YAML scenario and exercise the full loop:

```bash
docker compose exec app python scripts/demo_run.py --scenario examples/scenario.yaml
```

## Development

The source code lives under `src/`.  Each module has minimal stub implementations that you can extend.  Tests are located in `tests/` and can be run with `pytest`:

```bash
pytest -q
```

## File Structure

See `docs/arch.md` for a detailed architecture specification and folder overview.