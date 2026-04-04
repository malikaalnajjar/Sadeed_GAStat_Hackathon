# Step 2 — Start Sadeed

## 2.1 Create Your .env File

From the project root:

```bash
cd /home/aaldharrab/Projects/Sadeed

cat > .env << 'EOF'
REDIS_HOST=redis
REDIS_PORT=6379
OLLAMA_BASE_URL=http://ollama:11434
OLLAMA_MODEL=qwen3.5:9b
EOF
```

## 2.2 Start with Docker Compose

```bash
docker compose up --build
```

This starts three containers:

| Container | Port | Purpose |
|-----------|------|---------|
| `redis` | 6379 | Result caching |
| `ollama` | 11434 | LLM inference (GPU) |
| `backend` | 8000 | FastAPI server |

First run takes a few minutes — Ollama downloads the model inside its container and the backend warms it up.

## 2.3 Verify It's Running

Open a new terminal:

```bash
# Health check
curl http://localhost:8000/anomalies/health
```

Expected:

```json
{
  "great_expectations": true,
  "svm": true,
  "llm": true,
  "redis": true
}
```

Test a detection:

```bash
curl -s -X POST http://localhost:8000/anomalies/detect \
  -H "Content-Type: application/json" \
  -d '{
    "record_id": "test-1",
    "data": {
      "age": 12,
      "gender": 1600001,
      "family_relation": 1700001,
      "marage_status": 10600002,
      "nationality": 1800001,
      "q_301": 10500025,
      "q_602_val": 650000,
      "cut_5_total": 120,
      "act_1_total": 120
    }
  }' | python3 -m json.tool
```

You should see `"is_anomaly": true` with an Arabic explanation.

## 2.4 Running in the Background

Once everything works, restart in detached mode:

```bash
# Stop the foreground process (Ctrl+C), then:
docker compose up -d --build
```

Check logs anytime:

```bash
docker compose logs -f backend
```

Move on to [Step 3 — Install Cloudflare Tunnel](03-cloudflare-tunnel.md).
