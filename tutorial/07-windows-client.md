# Step 7 — Using Sadeed from Windows

Your Fedora laptop runs the server. This guide covers accessing it from a Windows machine.

## 7.1 Chrome Extension

### Install the Extension

1. Copy the `extension/` folder to the Windows machine (USB, cloud drive, git clone, etc.)
2. Open Chrome and go to `chrome://extensions/`
3. Enable **Developer mode** (top-right toggle)
4. Click **Load unpacked** and select the `extension/` folder
5. Pin the Sadeed icon in the toolbar

### Configure the Server URL

The extension defaults to `http://localhost:8000` which won't work on the Windows machine. You need to point it to your tunnel URL.

1. Click the Sadeed extension icon in the toolbar
2. Click the **gear icon** tab (settings)
3. Enter your server URL, for example:
   - Named tunnel: `https://sadeed.yourdomain.com`
   - Quick tunnel: `https://random-words-here.trycloudflare.com`
4. Click **Save**
5. The extension tests the connection automatically — a green dot in the header confirms it's working

No need to edit any code. The URL is saved in Chrome's sync storage and persists across restarts.

## 7.2 Test with PowerShell

Open PowerShell and run:

```powershell
# Health check
Invoke-RestMethod -Uri "https://sadeed.yourdomain.com/anomalies/health"

# Test detection
$body = @{
    record_id = "win-test-1"
    data = @{
        age = 12
        gender = 1600001
        family_relation = 1700001
        marage_status = 10600002
        nationality = 1800001
        q_301 = 10500025
        q_602_val = 650000
        cut_5_total = 120
        act_1_total = 120
    }
} | ConvertTo-Json

Invoke-RestMethod -Uri "https://sadeed.yourdomain.com/anomalies/detect" `
    -Method POST `
    -ContentType "application/json" `
    -Body $body
```

## 7.3 Test with curl (Windows)

curl comes pre-installed on Windows 10+:

```cmd
curl -s -X POST https://sadeed.yourdomain.com/anomalies/detect ^
  -H "Content-Type: application/json" ^
  -d "{\"record_id\": \"win-test-2\", \"data\": {\"age\": 35, \"gender\": 1600001, \"family_relation\": 1700001, \"marage_status\": 10600002, \"nationality\": 1800001, \"q_301\": 10500025, \"q_602_val\": 15000, \"cut_5_total\": 40, \"act_1_total\": 40}}"
```

## 7.4 Browser Quick Test

Open this in any browser on the Windows machine:

```
https://sadeed.yourdomain.com/anomalies/health
```

If you see the JSON health response, everything is connected.

## 7.5 Troubleshooting Windows-Specific Issues

### "Connection timed out"

- Check that the Cloudflare Tunnel is running on the Fedora laptop
- Check that Sadeed containers are up: `docker compose ps` on the laptop
- Try the URL from your phone to rule out a Windows firewall issue

### Chrome extension shows red dot

- Open the extension popup and go to the gear tab — verify the URL is correct
- Open DevTools on the extension popup (right-click > Inspect) and check the Console for errors

### SSL certificate errors

Cloudflare Tunnel handles HTTPS automatically. If you see certificate errors:
- Make sure you're using `https://` not `http://`
- If using a quick tunnel, the URL may have expired — restart cloudflared on the laptop

### Slow responses

Expected latency over the tunnel:
- Health check: < 200ms
- Detection (no LLM): < 500ms
- Detection (with LLM): 2-5s

If significantly slower, check your laptop's internet upload speed — the LLM response payload is small, so even 5 Mbps upload is fine.
