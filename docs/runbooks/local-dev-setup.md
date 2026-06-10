# Runbook: Local Development Setup

## Prerequisites
- Python 3.11+
- Git
- DVC: pip install dvc

## Steps

### 1. Clone and install
```
git clone https://github.com/Navin0245/scf-surrogate.git
cd scf-surrogate
pip install -e ".[dev]"
```

### 2. Pull data
```
dvc pull
```

### 3. Verify
```
pytest tests/unit/ -v
```

### 4. Run API locally
```
docker-compose up
```
