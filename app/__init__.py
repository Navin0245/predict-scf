"""FastAPI serving layer for the SCF surrogate.

Routes:
    POST /predict -- predict SCF for a joint geometry + load case
    GET  /health  -- liveness probe
    GET  /ready   -- readiness probe (model loaded?)
    GET  /docs    -- OpenAPI documentation (auto-generated)
"""
