# Changelog

## [0.1.0] - Initial implementation

### Added

- Architecture specification in `docs/arch.md`.
- Project skeleton with folders for source code (`src`), tests (`tests`), templates (`prompt_templates`), scripts (`scripts`), and infrastructure (`infra`).
- Stub implementations for core modules:
  - `config.py` using Pydantic `BaseSettings` for unified configuration.
  - LLM adapters (`llm/base.py`, `llm/openai.py`).
  - Stores (`store/pref_store.py`, `store/vector_store.py`, `store/feature_store.py`, `store/kv.py`).
  - Prompt compiler, loader, and linter (`prompt/compiler.py`, `prompt/loader.py`, `prompt/linter.py`).
  - Policy modules (`policy/bandit.py`, `policy/updater.py`).
  - Evaluation modules (`eval/heuristics.py`, `eval/eval_llm.py`).
  - Services (`services/orchestrator.py`, `services/guardrails.py`, `services/api.py`).
  - Utilities (`utils/embeddings.py`).
  - Scripts (`scripts/demo_run.py`, `scripts/ab_test.py`).
- Basic unit tests for prompt compiler, policy updater, and bandit skeleton.
- Dockerfile and docker-compose configuration for local development with PostgreSQL and Redis.