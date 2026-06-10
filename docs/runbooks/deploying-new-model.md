# Runbook: Deploying a New Model Version

## Steps

### 1. Verify quality gate
```
cat metrics/test_metrics.json
# MAE must be < 0.15
```

### 2. Register in MLflow
```
python scripts/register_model.py --run-id <id> --name scf-surrogate
```

### 3. Update MODEL_VERSION in configs/api_config.yaml

### 4. Redeploy
```
docker-compose up -d --build app
```

### 5. Verify
```
curl http://localhost:8000/health
```
