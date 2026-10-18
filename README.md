# MARZ backend

Flask + SQLite + OpenRouter. Python 3.11.

## Run locally
1. `pip install -r requirements.txt`
2. Copy `.env.example` to `.env` and set OPENROUTER_API_KEY.
3. `python check.py` runs a self-test with a fake model (no key or network needed).
4. `python app.py` starts the API on port 5000.

## Deploy on Render
Push this folder as its own repo, create a Web Service from render.yaml (or set
build `pip install -r requirements.txt`, start `gunicorn app:app --workers 1 --threads 4 --timeout 90`),
then set OPENROUTER_API_KEY and ALLOWED_ORIGINS (your Netlify URL, no trailing slash).

## Good to know
- Render's free disk is wiped on restart, so SQLite data does not persist there.
- The free server sleeps when idle; the first request can take about 30 seconds.
- openrouter/auto can route to paid models. Set MARZ_MODEL to a :free model to avoid charges.
