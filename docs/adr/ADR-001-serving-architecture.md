# ADR-001: Serving Architecture

## Status
Proposed

## Context
The SCF surrogate must be callable from SACS post-processing scripts.
Two options: (1) pip-installable library, (2) FastAPI microservice.

## Decision
FastAPI microservice.

## Consequences
+ Model version updates do not require reinstalling the library in SACS.
+ HTTP API is language-agnostic: SACS, Matlab, Excel can all call it.
+ Serving layer is independently deployable.
+ DNV validation enforced at network boundary, not in caller scripts.
- Network latency ~5ms per call vs direct function call.
  Acceptable: target <50ms, FEM baseline is 4-8 hours.

## Alternatives considered
pip library: rejected. Model versioning becomes caller responsibility.
gRPC: more efficient for high throughput but REST/JSON sufficient here.
