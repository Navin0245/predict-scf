# System Architecture

## Two-layer system
1. Offline training pipeline (Pipe-and-Filter)
2. Online serving layer (Microservice)

## Dependency rule
Dependencies point INWARD only:
    app/       --> src/
    pipelines/ --> src/
    src/       --> (nothing in this repo)

## Key decisions
See docs/adr/ for all architectural decisions with full context.
