# Step 6 — Troubleshooting

## Ollama: Model not loading / GPU not detected

```bash
# Check if Ollama container sees the GPU
docker compose exec ollama nvidia-smi

# Check Ollama logs
docker compose logs ollama
```

**Fix**: Make sure NVIDIA Container Toolkit is installed (see [Step 1.1](01-prerequisites.md#11-nvidia-container-toolkit)).

If the GPU is detected but the model won't load, it might be a VRAM issue:

```bash
# Check VRAM usage
nvidia-smi

# Kill other GPU processes if needed
sudo fuser -v /dev/nvidia*
```

## Backend: "LLM warmup failed"

This means the backend started before Ollama finished loading the model. It's non-fatal — the LLM will work once Ollama is ready.

```bash
# Check if Ollama is responding
curl http://localhost:11434/api/tags

# Manually trigger warmup
curl -X POST http://localhost:11434/api/generate \
  -d '{"model": "qwen3.5:9b", "prompt": "ping", "stream": false}'
```

## Cloudflare Tunnel: Connection refused

```bash
# Is the tunnel running?
systemctl status cloudflared

# Check tunnel logs
journalctl -u cloudflared -f

# Is the backend reachable locally?
curl http://localhost:8000/health
```

**Common causes**:
- Backend not running — start Sadeed first, then the tunnel
- Wrong port in `config.yml` — must be `http://localhost:8000`
- Firewall blocking local connections — shouldn't be an issue since the tunnel connects outbound

## Docker Compose: Port already in use

```bash
# Find what's using port 8000
sudo ss -tlnp | grep 8000

# Or Redis port 6379
sudo ss -tlnp | grep 6379
```

**Fix**: Stop the conflicting process, or change ports in `docker-compose.yml`.

## Slow LLM Responses

Expected latency on your RTX 5080:
- First request after startup: 5-10s (model loading into VRAM)
- Subsequent requests: 1-3s

If responses are taking 30s+, the model might be running on CPU:

```bash
# Check if Ollama is using GPU
docker compose exec ollama nvidia-smi
```

If no GPU usage shows, restart with GPU access:

```bash
docker compose down
docker compose up -d
```

## Redis: Connection refused

```bash
# Is Redis running?
docker compose ps redis

# Check Redis logs
docker compose logs redis

# Test Redis directly
docker compose exec redis redis-cli ping
```

Expected: `PONG`

## Full Reset

If everything is broken, start fresh:

```bash
# Stop and remove all containers + volumes
docker compose down -v

# Rebuild from scratch
docker compose up --build -d

# Watch logs until healthy
docker compose logs -f
```

## Health Check Script

Save this as a quick diagnostic:

```bash
#!/bin/bash
echo "=== Docker Containers ==="
docker compose ps

echo -e "\n=== GPU Status ==="
nvidia-smi --query-gpu=name,memory.used,memory.total,temperature.gpu --format=csv,noheader

echo -e "\n=== Backend Health ==="
curl -s http://localhost:8000/anomalies/health | python3 -m json.tool 2>/dev/null || echo "Backend not reachable"

echo -e "\n=== Tunnel Status ==="
systemctl is-active cloudflared 2>/dev/null || echo "Tunnel not running as service"
```
