# Step 5 — Keep It Running

## 5.1 Auto-Start Docker Compose on Boot

Create a systemd service for Sadeed:

```bash
sudo tee /etc/systemd/system/sadeed.service << 'EOF'
[Unit]
Description=Sadeed Anomaly Detection
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/home/aaldharrab/Projects/Sadeed
ExecStart=/usr/bin/docker compose up -d
ExecStop=/usr/bin/docker compose down
User=aaldharrab

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable sadeed.service
```

Now Sadeed starts automatically on boot.

Manual control:

```bash
sudo systemctl start sadeed    # Start
sudo systemctl stop sadeed     # Stop
sudo systemctl status sadeed   # Check status
```

## 5.2 Auto-Start Cloudflare Tunnel on Boot

### For Named Tunnels

```bash
sudo cloudflared service install
sudo systemctl enable cloudflared
sudo systemctl start cloudflared
```

This runs the tunnel as a system service using `~/.cloudflared/config.yml`.

### For Quick Tunnels

Quick tunnels can't be installed as a service (URL changes each time). Use a named tunnel for persistent hosting.

## 5.3 Laptop Power Settings

Since this is a laptop acting as a server:

**Prevent sleep on lid close** (KDE Plasma):

1. System Settings > Power Management > On AC Power
2. Set "When laptop lid closed" to **Do nothing**

Or via config:

```bash
# Prevent sleep on lid close when on AC power
sudo mkdir -p /etc/systemd/logind.conf.d
sudo tee /etc/systemd/logind.conf.d/lid.conf << 'EOF'
[Login]
HandleLidSwitch=ignore
HandleLidSwitchExternalPower=ignore
EOF

sudo systemctl restart systemd-logind
```

**Keep Wi-Fi alive during sleep** (if applicable):

```bash
# Disable Wi-Fi power saving
sudo nmcli connection modify "YOUR_WIFI_NAME" 802-11-wireless.powersave 2
```

Replace `YOUR_WIFI_NAME` with your connection name (check with `nmcli connection show`).

## 5.4 Monitor Resource Usage

Keep an eye on GPU memory and temperature:

```bash
# Live GPU stats
watch -n 2 nvidia-smi

# Docker container health
docker compose ps
docker compose logs -f --tail=50 backend

# Quick health check
curl -s http://localhost:8000/anomalies/health | python3 -m json.tool
```

## 5.5 Summary — What Starts on Boot

| Service | Method | Command to Check |
|---------|--------|-----------------|
| Docker | Enabled by default | `systemctl status docker` |
| Sadeed (Redis + Ollama + Backend) | systemd unit | `systemctl status sadeed` |
| Cloudflare Tunnel | cloudflared service | `systemctl status cloudflared` |

After a reboot, everything should come back up automatically within about 1-2 minutes (Ollama needs to reload the model into VRAM).
