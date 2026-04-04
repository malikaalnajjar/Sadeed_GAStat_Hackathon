# Step 1 — Prerequisites

## 1.1 NVIDIA Container Toolkit

Docker needs GPU access to run Ollama with CUDA. Check if it's already set up:

```bash
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu24.04 nvidia-smi
```

If that prints your GPU info, skip to [1.2](#12-redis). Otherwise, install the toolkit:

```bash
# Add NVIDIA container toolkit repo
curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo | \
  sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo

# Install
sudo dnf install -y nvidia-container-toolkit

# Configure Docker runtime
sudo nvidia-ctk runtime configure --runtime=docker

# Restart Docker
sudo systemctl restart docker
```

Verify it works:

```bash
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu24.04 nvidia-smi
```

You should see your RTX 5080 with 16GB VRAM.

## 1.2 Redis

Redis is pulled automatically by Docker Compose — no manual install needed.

## 1.3 Ollama Model

If you haven't pulled the model yet:

```bash
ollama pull qwen3.5:9b
```

This downloads ~5.5GB. Your 16GB VRAM handles this comfortably.

## 1.4 Verify Everything

Run this quick check:

```bash
# GPU accessible?
nvidia-smi

# Docker running?
docker info > /dev/null 2>&1 && echo "Docker: OK" || echo "Docker: NOT RUNNING"

# Ollama installed?
ollama --version

# Model available?
ollama list | grep qwen3.5
```

All green? Move on to [Step 2 — Start Sadeed](02-start-sadeed.md).
