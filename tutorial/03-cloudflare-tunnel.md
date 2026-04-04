# Step 3 — Install Cloudflare Tunnel

Cloudflare Tunnel creates an encrypted outbound connection from your laptop to Cloudflare's edge network. No open ports on your router, free HTTPS, and your home IP stays hidden.

## 3.1 Install cloudflared

```bash
# Fedora
sudo dnf install -y cloudflared
```

If not available in your repos, download directly:

```bash
# Download the latest binary
curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-x86_64 \
  -o /tmp/cloudflared

# Install it
sudo install -m 755 /tmp/cloudflared /usr/local/bin/cloudflared

# Verify
cloudflared --version
```

## 3.2 Authenticate with Cloudflare

```bash
cloudflared tunnel login
```

This opens a browser window. Log in to your Cloudflare account and authorize.

### Don't have a Cloudflare account?

1. Go to https://dash.cloudflare.com/sign-up (free)
2. Sign up with email
3. Run `cloudflared tunnel login` again

### Don't have a domain?

You have two options:

**Option A: Quick tunnel (no domain needed)** — Gives you a random `*.trycloudflare.com` URL. Great for testing but the URL changes every restart. Skip to [Step 4 Quick Tunnel](04-expose.md#option-a-quick-tunnel-no-domain).

**Option B: Named tunnel with your own domain** — Stable URL, persists across restarts. You need a domain added to Cloudflare (even a cheap $5-10/year `.xyz` or `.site` domain works). Continue to [3.3](#33-create-a-named-tunnel).

## 3.3 Create a Named Tunnel

```bash
cloudflared tunnel create sadeed
```

This outputs a tunnel UUID. Note it down:

```
Created tunnel sadeed with id a1b2c3d4-e5f6-...
```

## 3.4 Configure DNS

Point a subdomain at the tunnel:

```bash
cloudflared tunnel route dns sadeed sadeed.yourdomain.com
```

Replace `yourdomain.com` with your actual domain.

Verify in the Cloudflare dashboard: **DNS > Records** — you should see a new CNAME record for `sadeed` pointing to `<tunnel-id>.cfargotunnel.com`.

## 3.5 Create the Tunnel Config File

```bash
mkdir -p ~/.cloudflared

cat > ~/.cloudflared/config.yml << 'EOF'
tunnel: sadeed
credentials-file: /home/aaldharrab/.cloudflared/<TUNNEL_ID>.json

ingress:
  - hostname: sadeed.yourdomain.com
    service: http://localhost:8000
  - service: http_status:404
EOF
```

Replace:
- `<TUNNEL_ID>` with the UUID from step 3.3
- `sadeed.yourdomain.com` with your actual subdomain

Move on to [Step 4 — Expose to the Internet](04-expose.md).
