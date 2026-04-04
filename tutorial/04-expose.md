# Step 4 — Expose to the Internet

Make sure Sadeed is running first (`docker compose up -d`).

## Option A: Quick Tunnel (No Domain)

One command, no config needed:

```bash
cloudflared tunnel --url http://localhost:8000
```

Output:

```
Your quick Tunnel has been created! Visit it at:
https://random-words-here.trycloudflare.com
```

That URL is live. Test it:

```bash
curl https://random-words-here.trycloudflare.com/health
```

**Limitations**: URL changes every time you restart cloudflared. Good for demos and testing.

## Option B: Named Tunnel (Stable URL)

Requires the setup from [Step 3.3-3.5](03-cloudflare-tunnel.md#33-create-a-named-tunnel).

```bash
cloudflared tunnel run sadeed
```

Your API is now live at `https://sadeed.yourdomain.com`.

Test it:

```bash
curl https://sadeed.yourdomain.com/anomalies/health
```

## Verify End-to-End

From any device (phone, another computer, etc.):

```bash
curl -s -X POST https://sadeed.yourdomain.com/anomalies/detect \
  -H "Content-Type: application/json" \
  -d '{
    "record_id": "remote-test",
    "data": {
      "age": 35,
      "gender": 1600001,
      "family_relation": 1700001,
      "marage_status": 10600002,
      "nationality": 1800001,
      "q_301": 10500025,
      "q_602_val": 15000,
      "cut_5_total": 40,
      "act_1_total": 40
    }
  }' | python3 -m json.tool
```

Move on to [Step 5 — Keep It Running](05-keep-running.md).
