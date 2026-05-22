These are all a work in progress and will be ever changing.  I will try and add dates with changes made to the files as I can.  TRADE AT YOUR OWN RISK USING THESE TOOLS AS THIS IS NOT FINANCIAL ADVICE!

The pinescripts can be deployed inside of tradingview free version as i do not pay for the paid version myself.

# CSP Options Scanner and DITM call leap scanner — Web App

Run the scanner from any browser, including your phone.

## Run Locally (on your computer)

```bash
# 1. Install dependencies (one time)
pip install -r requirements.txt

# 2. Start the server
python app.py

# 3. Open in browser
#    Desktop:  http://localhost:5000
#    Phone on same WiFi:  http://YOUR_COMPUTER_IP:5000
#    (find your IP: Windows → ipconfig | Mac → ifconfig | look for 192.168.x.x)
```

---

## Deploy Free to Railway (access from ANYWHERE — including phone on cell data)

Railway gives you a permanent URL like `https://csp-scanner.up.railway.app`
that works on any device, anywhere, no VPN needed.

### Steps (takes ~5 minutes)

1. **Create a free Railway account** at https://railway.app
   - Sign up with GitHub (easiest) — no credit card required

2. **Install Railway CLI** (optional but fastest):
   ```bash
   npm install -g @railway/cli
   railway login
   ```

3. **Create a new project from this folder**:
   ```bash
   cd csp_webapp
   railway init          # name it "csp-scanner"
   railway up            # deploys the app
   railway open          # opens your live URL
   ```

   **Or use the Railway web dashboard** (no CLI needed):
   - Go to https://railway.app/dashboard
   - Click "New Project" → "Deploy from GitHub repo"
   - Push this folder to a GitHub repo first, then connect it

4. **Add a Procfile** (Railway needs this to know how to start the app):
   ```
   web: python app.py
   ```
   Create a file named exactly `Procfile` (no extension) with that one line.

5. **Set the PORT environment variable** (Railway sets this automatically).
   The app already reads `os.environ.get("PORT", 5000)` so no changes needed.

6. **Access from your phone**: Railway gives you a URL like
   `https://csp-scanner-production.up.railway.app`
   — bookmark it or add it to your Android home screen.

---

## Add to Android Home Screen (feels like an app)

1. Open Chrome on your Android phone
2. Go to your Railway URL
3. Tap the ⋮ menu → "Add to Home screen"
4. Tap "Add"

It will appear on your home screen with an icon and open full-screen,
just like a native app.

---

## Alternative Free Hosts

| Host | Free Tier | Notes |
|------|-----------|-------|
| **Railway** | 500 hrs/month | Easiest, recommended |
| **Render** | 750 hrs/month | Sleeps after 15min inactivity |
| **PythonAnywhere** | Always on | Slower, but truly free forever |
| **Fly.io** | 3 shared VMs | More setup required |

---

## Usage

- **Home tab**: configure and launch a scan
- **Scanning tab**: live progress ring, ticker-by-ticker log
- **Results tab**: all qualifying contracts as cards, filterable and sortable
- **Export CSV**: downloads the current (filtered) results to your device

---

## Notes

- Run during NYSE market hours (Mon–Fri 9:30–16:00 ET) for accurate volume data
- Enable "Relax Filters" outside market hours to bypass the volume filter
- A full scan takes 15–30 minutes depending on workers and network speed
- The server keeps the last scan result in memory — restarting it clears results
