# scf-surrogate

Production ML surrogate for predicting Stress Concentration Factors (SCF)
in offshore tubular joints.

Replaces solid FEM analysis (4-8 hours) with a validated prediction API (<50ms).

Compliance: DNV-RP-C203 (2021) | API RP 2A-WSD (22nd Ed.)

## Quickstart

```bash
pip install -e ".[dev]"
dvc pull
pytest tests/unit/
docker-compose up
```

## Architecture

1. Offline training pipeline (Pipe-and-Filter, DVC-orchestrated)
2. Online serving layer (FastAPI microservice, MLflow model registry)

See docs/design/system-architecture.md

## Project structure

```
src/scf_surrogate/   -- domain logic (joints, equations)
pipelines/           -- data generation and model training
app/                 -- FastAPI prediction API
tests/               -- unit, integration, e2e
docs/                -- ADRs, runbooks, standards references
data/                -- DVC-managed (not in git)
```

## Key results

| Metric        | Value | Target              |
|---------------|-------|---------------------|
| SCF MAE       | TBD   | < 15% (DNV)         |
| Latency p99   | TBD   | < 50ms              |
| Test coverage | TBD   | >= 80%              |
