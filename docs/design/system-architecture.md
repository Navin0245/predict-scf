# SCF Surrogate — System Architecture

> **Project:** Offshore Tubular Joint SCF Prediction  
> **Compliance:** DNV-RP-C203 (2021) · API RP 2A-WSD (22nd Ed.)  
> **Version:** 0.1.0 · **Author:** Navin

---

## 1. System Overview

The SCF Surrogate is a two-layer ML system. The **offline layer** generates
data and trains the model. The **online layer** serves predictions through a
validated REST API. Both layers share a common domain logic library (`src/`).

The core engineering problem: replace 4–8 hour solid FEM analyses with
sub-50ms validated predictions for offshore jacket fatigue assessments.

---

## 2. ML vs Non-ML Boundary

The most important architectural boundary in any ML system. Everything inside
the ML boundary is statistical and approximate. Everything outside is
deterministic and must be exact.

```mermaid
flowchart TB
    classDef nonml  fill:#e8f5e9,stroke:#2e7d32,stroke-width:1.5px,color:#1b5e20
    classDef ml     fill:#e3f2fd,stroke:#1565c0,stroke-width:1.5px,color:#0d47a1
    classDef infra  fill:#fce4ec,stroke:#880e4f,stroke-width:1.5px,color:#880e4f
    classDef data   fill:#fff8e1,stroke:#f57f17,stroke-width:1.5px,color:#e65100

    subgraph NML["Non-ML  —  deterministic, exact"]
        A[FastAPI Endpoint]:::nonml
        B[Pydantic Validator\nDNV bounds enforced]:::nonml
        C[Request Router]:::nonml
        D[structlog\nObservability]:::nonml
        E[GitHub Actions\nCI / CD]:::nonml
        F[pytest Suite\nContinuous testing]:::nonml
    end

    subgraph ML["ML  —  statistical, approximate"]
        G[taxonomy.py\nFeature Encoder]:::ml
        H[load_classification.py\nLoad Fraction Encoder]:::ml
        I[GNN / XGBoost\nSurrogate Model]:::ml
        J[SCF Prediction\nAssembler]:::ml
    end

    subgraph INF["Infrastructure"]
        K[(MLflow\nModel Registry)]:::infra
        L[(DVC\nData Store)]:::infra
        M[Model Loader\nStartup Event]:::infra
    end

    A --> B --> C --> G
    G --> H --> I --> J
    J --> A
    K --> M --> I
    A --> D
    E --> F

    style NML fill:#f1f8e9,stroke:#558b2f,stroke-width:2px
    style ML  fill:#e8eaf6,stroke:#3949ab,stroke-width:2px
    style INF fill:#fce4ec,stroke:#c62828,stroke-width:2px
```

> **Rule:** Dependencies point inward only.  
> `app/` → `src/` · `pipelines/` → `src/` · `src/` → nothing in this repo.

---

## 3. Offline Training Pipeline — Pipe-and-Filter Pattern

Each stage is a **pure filter**: typed input → typed output, independently
testable, DVC-tracked. If only one filter changes, only that filter and
its downstream stages re-run. Upstream results are cached by content hash.

```mermaid
flowchart LR
    classDef filter fill:#fff3e0,stroke:#e65100,stroke-width:1.5px,color:#bf360c
    classDef store  fill:#e8eaf6,stroke:#3949ab,stroke-width:1.5px,color:#1a237e
    classDef model  fill:#e3f2fd,stroke:#1565c0,stroke-width:1.5px,color:#0d47a1

    F1([" 1 · Sampler\nLatin Hypercube\nN = 10 000 "]):::filter
    F2([" 2 · DNV Validator\ntaxonomy.py\nRejects out-of-range "]):::filter
    F3([" 3 · Efthymiou Labeller\nSCF per brace\nper load location "]):::filter
    F4([" 4 · Load Classifier\nload_classification.py\nf_K · f_TY · f_cross "]):::filter
    F5([" 5 · Feature Encoder\nStandardScaler\nFixed-length vector "]):::filter
    F6([" 6 · Splitter\n70 / 15 / 15\nStratified by joint type "]):::filter
    TR([" 7 · Trainer\nGNN / XGBoost\nMLflow tracked "]):::model

    S1[(raw_params\n.parquet)]:::store
    S2[(validated\n.parquet)]:::store
    S3[(labelled\n.parquet)]:::store
    S4[(classified\n.parquet)]:::store
    S5[(features\n.parquet)]:::store
    S6[(train · val · test\n.parquet)]:::store
    S7[(model\nartifact)]:::store

    F1 --> S1 --> F2 --> S2 --> F3 --> S3 --> F4
    F4 --> S4 --> F5 --> S5 --> F6 --> S6 --> TR --> S7
```

**Data quality invariants enforced at every store boundary:**
- No SCF value < 1.0 (physical impossibility — concentration not reduction)
- No parameter outside DNV validity range enters the model
- No NaN in any feature column
- Train + Val + Test = 100% of validated samples

---

## 4. Online Serving Layer — Microservice Pattern

The serving layer is **stateless**: it loads a frozen model artifact at
startup and answers HTTP requests. It has no knowledge of training code.
Model version updates do not require redeployment — only a config change.

```mermaid
flowchart TB
    classDef client  fill:#f5f5f5,stroke:#757575,stroke-width:1.5px,color:#212121
    classDef route   fill:#e8f5e9,stroke:#2e7d32,stroke-width:1.5px,color:#1b5e20
    classDef schema  fill:#e0f2f1,stroke:#00695c,stroke-width:1.5px,color:#004d40
    classDef domain  fill:#e3f2fd,stroke:#1565c0,stroke-width:1.5px,color:#0d47a1
    classDef infra   fill:#fce4ec,stroke:#880e4f,stroke-width:1.5px,color:#880e4f

    CLIENT["SACS Script · Postman · pytest"]:::client

    subgraph GW["FastAPI — app/"]
        R1["POST /predict"]:::route
        R2["GET  /health"]:::route
        R3["GET  /ready"]:::route
        R4["GET  /docs\nOpenAPI auto-generated"]:::route
        V["JointGeometryRequest\nPydantic · DNV bounds"]:::schema
        RS["SCFPredictionResponse\nPydantic · typed output"]:::schema
    end

    subgraph DOM["Domain — src/scf_surrogate/"]
        TX["taxonomy.py\nto_feature_vector()"]:::domain
        LC["load_classification.py\ninterpolated_scf()"]:::domain
        EX["exceptions.py\nParameterRangeError\nMissingGapError"]:::domain
    end

    subgraph INF["Infrastructure"]
        ML["MLflow Registry\nmodel artifact"]:::infra
        LDR["Model Loader\nstartup event"]:::infra
        LOG["structlog\nJSON logs"]:::infra
    end

    CLIENT -->|"POST JSON"| R1
    CLIENT -->|"GET"| R2
    CLIENT -->|"GET"| R3
    R1 --> V
    V -->|"422 if invalid"| CLIENT
    V --> TX --> LC --> LDR
    LDR -->|"predict()"| RS
    RS -->|"200 JSON"| CLIENT
    ML --> LDR
    R1 --> LOG
    R2 --> LOG

    style GW  fill:#f1f8e9,stroke:#558b2f,stroke-width:2px
    style DOM fill:#e8eaf6,stroke:#3949ab,stroke-width:2px
    style INF fill:#fce4ec,stroke:#c62828,stroke-width:2px
```

---

## 5. Request Lifecycle — End to End

One prediction request, traced through every component.

```mermaid
sequenceDiagram
    autonumber
    actor       C  as SACS Script
    participant F  as FastAPI
    participant P  as Pydantic Validator
    participant T  as taxonomy.py
    participant L  as load_classification.py
    participant M  as SCF Model
    participant G  as structlog

    C  ->>  F : POST /predict {beta, gamma, tau, theta, f_k, f_ty, f_cross}

    F  ->>  P : validate(request_body)

    alt DNV bounds violated
        P -->> F : ParameterRangeError
        F -->> C : 422 Unprocessable Entity
        F ->>  G : WARN  validation_failed  {param, value, range}
    else valid geometry
        P -->> F : JointGeometryRequest  ✓
        F  ->>  T : geom.to_feature_vector()
        T -->> F : [alpha, beta, gamma, tau, theta_rad, is_TY…, zeta]
        F  ->>  L : BraceLoadClassification(f_k, f_ty, f_cross)
        L -->> F : validated classification
        F  ->>  M : model.predict(feature_vector)
        M -->> F : {scf_k, scf_ty, scf_cross}
        F  ->>  L : interpolated_scf(scf_k, scf_ty, scf_cross, clf)
        L -->> F : SCF = 10.86
        F -->> C : 200 OK  {scf, dominant_mode, is_within_dnv, model_version}
        F  ->>  G : INFO  prediction_served  {latency_ms, joint_type, scf}
    end
```

---

## 6. CI/CD Contract — What Each Gate Enforces

Each job in the pipeline is a **contract**, not just a step.
The order is deliberate: cheapest checks first, most expensive last.

```mermaid
flowchart TD
    classDef gate    fill:#e8f5e9,stroke:#2e7d32,stroke-width:1.5px,color:#1b5e20
    classDef mlgate  fill:#e3f2fd,stroke:#1565c0,stroke-width:1.5px,color:#0d47a1
    classDef deploy  fill:#f3e5f5,stroke:#6a1b9a,stroke-width:1.5px,color:#4a148c
    classDef block   fill:#ffebee,stroke:#c62828,stroke-width:1.5px,color:#b71c1c
    classDef trigger fill:#fff8e1,stroke:#f57f17,stroke-width:1.5px,color:#e65100

    T([" git push / PR opened "]):::trigger

    G1["ruff check\nStyle · imports · security\n~5 s"]:::gate
    G2["mypy --strict\nType safety\n~15 s"]:::gate
    G3["pytest unit\ntaxonomy · load_classification\n~10 s"]:::gate
    G4["pytest integration\nPipeline filters end-to-end\n~60 s"]:::gate
    G5["Coverage ≥ 80 %"]:::gate
    G6["dvc repro\nRebuild dataset + model\n~10 min"]:::mlgate
    G7["MAE < 15 %\nDNV acceptance criterion"]:::mlgate
    G8["docker build\nMulti-stage image"]:::deploy
    G9["Push to registry\nTagged with git SHA"]:::deploy

    B1([" BLOCK — fix locally "]):::block
    B2([" BLOCK — coverage below threshold "]):::block
    B3([" BLOCK — model quality degraded "]):::block

    T --> G1 --> G2
    G2 -->|fail| B1
    G2 --> G3 --> G4 --> G5
    G5 -->|fail| B2
    G5 --> G6 --> G7
    G7 -->|fail| B3
    G7 --> G8 --> G9
```

> **Why this order:**  
> A type error caught by mypy in 15 s avoids wasting 60 s on tests that
> will fail anyway. The ML quality gate runs last because retraining is
> expensive — only pay that cost after all cheap correctness checks pass.

---

## 7. Repository Structure

```
scf-surrogate/
│
├── src/scf_surrogate/          ← domain logic (no ML framework)
│   ├── exceptions.py           ← central exception hierarchy
│   └── joints/
│       ├── taxonomy.py         ← JointType · TubularJointGeometry · validate_dnv_bounds
│       └── load_classification.py  ← BraceLoadClassification · interpolated_scf
│
├── pipelines/                  ← offline: data generation + training
│   ├── filters/                ← one module per Pipe-and-Filter stage
│   └── training/               ← GNN/XGBoost trainer + evaluator
│
├── app/                        ← online: FastAPI serving layer
│   ├── schemas/                ← Pydantic request + response models
│   └── routes/                 ← /predict · /health · /ready
│
├── tests/
│   ├── unit/                   ← fast, no I/O, milliseconds
│   ├── integration/            ← component interactions, seconds
│   └── e2e/                    ← full pipeline, minutes (CI only)
│
├── docs/
│   ├── adr/                    ← Architecture Decision Records
│   ├── standards/              ← DNV · API RP 2A summaries
│   └── runbooks/               ← deploy · dev setup · incident response
│
├── data/                       ← DVC-managed (not in git)
├── models/                     ← DVC-managed (not in git)
├── metrics/                    ← JSON metrics (in git — tiny files)
├── configs/                    ← YAML configuration (no hardcoded values)
├── .github/workflows/          ← ci.yml · cd.yml
├── Dockerfile                  ← multi-stage build
└── docker-compose.yml          ← API + MLflow for local dev
```

---

## 8. Data Versioning Strategy

```mermaid
flowchart LR
    classDef git   fill:#e8f5e9,stroke:#2e7d32,stroke-width:1.5px,color:#1b5e20
    classDef dvc   fill:#e3f2fd,stroke:#1565c0,stroke-width:1.5px,color:#0d47a1

    subgraph GIT["git tracks  —  small files"]
        A["pyproject.toml"]:::git
        B["dvc.yaml\npipeline DAG"]:::git
        C["data/raw/params.dvc\n5-line pointer"]:::git
        D["metrics/val_metrics.json\nMAE · RMSE"]:::git
        E[".github/workflows/\nCI · CD YAML"]:::git
    end

    subgraph DVC["DVC remote tracks  —  large files"]
        F["data/raw/params.parquet\n~50 MB"]:::dvc
        G["data/splits/train.parquet"]:::dvc
        H["models/scf_surrogate_v1.pkl"]:::dvc
        I["reports/scf_error_plots.png"]:::dvc
    end

    C -->|"content hash\npoints to"| F

    style GIT fill:#f1f8e9,stroke:#558b2f,stroke-width:2px
    style DVC fill:#e8eaf6,stroke:#3949ab,stroke-width:2px
```

**Why `metrics/` goes in git but `data/` does not:**  
Metrics are tiny JSON files — a few hundred bytes. Committing them means
`git log metrics/val_metrics.json` shows exactly how model quality changed
with every commit. This is how you write the results section of your paper.

---

## 9. Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Serving pattern | Microservice (FastAPI) | Language-agnostic HTTP interface; SACS, Matlab, Excel can all call it |
| Pipeline pattern | Pipe-and-Filter (DVC) | Each filter independently testable; reproducibility structural not procedural |
| Validation boundary | Pydantic at API entry | DNV bounds enforced once, at the network boundary, not scattered in callers |
| Data generation | Efthymiou equations | Analytical ground truth; no historical data required |
| Model first choice | XGBoost → GNN | XGBoost establishes interpretable baseline; GNN captures topology effects |
| Reproducibility | DVC + MLflow + git SHA | Every prediction traceable to exact data version, code version, run ID |
| Exception hierarchy | Central `exceptions.py` | All domain errors inherit from `SCFSurrogateError`; callers catch one base type |

---

*See `docs/adr/` for full Architecture Decision Records with context and alternatives considered.*