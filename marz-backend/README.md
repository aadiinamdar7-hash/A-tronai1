# MARZ backend
Flask + SQLite + OpenRouter API.

Local: copy `.env.example` to `.env`, add OPENROUTER_API_KEY, then
`pip install -r requirements.txt && python app.py` (runs on port 5000).

Deploy: push this folder to Render (render.yaml included). Set OPENROUTER_API_KEY and
ALLOWED_ORIGINS (your Netlify URL, e.g. https://marz.netlify.app).
Note: Render's free disk is wiped on restart, so the SQLite data does not persist there.
