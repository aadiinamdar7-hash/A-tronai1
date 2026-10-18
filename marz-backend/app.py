"""MARZ backend: Flask + SQLite + OpenRouter."""
import os, re, time, logging, sqlite3
import requests
from dotenv import load_dotenv
from flask import Flask, request, jsonify, g, abort
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.exceptions import HTTPException
from werkzeug.middleware.proxy_fix import ProxyFix

load_dotenv()
KEY = os.environ.get("OPENROUTER_API_KEY")
MODEL = os.environ.get("MARZ_MODEL", "openrouter/auto")
DB = os.environ.get("DB_PATH", "marz.db")
ORIGINS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "http://localhost:8080").split(",")]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("marz")

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1)
app.config["MAX_CONTENT_LENGTH"] = 6 * 1024 * 1024
CORS(app, origins=ORIGINS, allow_headers=["Content-Type", "X-User-ID"])
limiter = Limiter(get_remote_address, app=app, default_limits=["300/hour"], storage_uri="memory://")

SCHEMA = """
CREATE TABLE IF NOT EXISTS users(id TEXT PRIMARY KEY, created REAL);
CREATE TABLE IF NOT EXISTS memories(id INTEGER PRIMARY KEY, user_id TEXT, kind TEXT DEFAULT 'manual', content TEXT, created REAL, UNIQUE(user_id, content));
CREATE TABLE IF NOT EXISTS conversations(id INTEGER PRIMARY KEY, user_id TEXT, title TEXT, mode TEXT, created REAL);
CREATE TABLE IF NOT EXISTS messages(id INTEGER PRIMARY KEY, conv_id INTEGER, role TEXT, content TEXT, created REAL);
CREATE TABLE IF NOT EXISTS projects(id INTEGER PRIMARY KEY, user_id TEXT, name TEXT, description TEXT, created REAL);
CREATE TABLE IF NOT EXISTS settings(user_id TEXT, key TEXT, value TEXT, PRIMARY KEY(user_id, key));
CREATE TABLE IF NOT EXISTS saved_responses(id INTEGER PRIMARY KEY, user_id TEXT, content TEXT, created REAL);
CREATE TABLE IF NOT EXISTS notes(id INTEGER PRIMARY KEY, user_id TEXT, title TEXT, body TEXT, created REAL);
CREATE INDEX IF NOT EXISTS idx_msg ON messages(conv_id);
"""
with sqlite3.connect(DB) as _c:
    _c.execute("PRAGMA journal_mode=WAL")
    _c.executescript(SCHEMA)


def db():
    if "db" not in g:
        g.db = sqlite3.connect(DB, timeout=10)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(_):
    d = g.pop("db", None)
    if d:
        d.close()


def q(sql, a=(), w=False):
    cur = db().execute(sql, a)
    if w:
        db().commit()
        return cur.lastrowid
    return [dict(r) for r in cur.fetchall()]


def user():
    u = request.headers.get("X-User-ID", "")
    if not re.fullmatch(r"[A-Za-z0-9-]{16,64}", u):
        abort(401, "Missing or invalid X-User-ID.")
    q("INSERT OR IGNORE INTO users(id, created) VALUES(?,?)", (u, time.time()), w=True)
    return u


@app.errorhandler(Exception)
def fail(e):
    if isinstance(e, HTTPException):
        return jsonify(error=e.description), e.code
    log.exception("Unhandled error")
    return jsonify(error="Something went wrong on the server."), 500


# ---------- modes & prompts ----------
BASE = ("You are MARZ, a capable, warm and concise personal AI assistant with a calm, JARVIS-like poise. "
        "Use British English. Be accurate and say when you are unsure.")
MODES = {
    "general": "Answer helpfully and conversationally.",
    "study": "Act as a patient tutor: explain simply, check understanding, and offer summaries, flashcards (Q/A) and exam-style questions.",
    "code": "Act as a senior engineer: give working, minimal code in fenced blocks, explain bugs, architecture and trade-offs.",
    "research": "Act as a research analyst: structured, balanced answers, comparisons, separate fact from uncertainty, and flag what to verify (you cannot browse the web).",
    "creative": "Act as a creative writing partner: vivid, original voice, offer variations.",
    "assistant": "Act as a personal assistant: turn requests into plans, schedules and checklists; ask at most one question.",
}
HINTS = {
    "code": r"```|\b(python|javascript|html|css|flask|bug|debug|function|stack ?trace|error|api|sql|code)\b",
    "study": r"\b(revise|revision|exam|flashcards?|quiz|gcse|a-level|homework|study|syllabus)\b",
    "research": r"\b(research|compare|versus|pros and cons|how does|why does|history of|evidence)\b",
    "creative": r"\b(story|poem|lyrics|novel|character|slogan|caption|brainstorm)\b",
    "assistant": r"\b(remind|schedule|plan my|to-?do|timetable|agenda|organi[sz]e)\b",
}


def detect(text, last):
    for m, p in HINTS.items():
        if re.search(p, text, re.I):
            return m
    return last


def build_system(u, mode):
    mem = q("SELECT content FROM memories WHERE user_id=? ORDER BY created DESC LIMIT 40", (u,))
    proj = q("SELECT name, description FROM projects WHERE user_id=? ORDER BY created DESC LIMIT 5", (u,))
    nm = q("SELECT value FROM settings WHERE user_id=? AND key='name'", (u,))
    s = f"{BASE}\nMode: {mode}. {MODES[mode]}"
    if nm and nm[0]["value"]:
        s += f"\nThe user's name is {nm[0]['value']}."
    if mem:
        s += "\nWhat you know about the user (use naturally, never recite):\n" + "\n".join("- " + m["content"] for m in mem)
    if proj:
        s += "\nActive projects:\n" + "\n".join(f"- {p['name']}: {p['description']}" for p in proj)
    s += ("\nIf the user reveals a durable fact worth remembering (studies, preferences, projects, goals, interests), "
          "end your reply with one line: <memory>short third-person fact</memory>. Never mention this tag.")
    return s


def llm(msgs):
    r = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {KEY}", "HTTP-Referer": os.environ.get("SITE_URL", "http://localhost"), "X-Title": "MARZ"},
        json={"model": MODEL, "messages": msgs, "max_tokens": 1500},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"] or ""


MEM_TAG = re.compile(r"<memory>(.*?)</memory>", re.S | re.I)
FACT = re.compile(r"^\s*(i am|i'm|i’m|my name is|my goal|i want to|i like|i love|i study)\b", re.I)


@app.post("/api/chat")
@limiter.limit("20/minute")
def chat():
    u = user()
    d = request.get_json(silent=True) or {}
    text = str(d.get("message", "")).strip()[:4000]
    img = d.get("image")
    if not text and not img:
        abort(400, "Empty message.")
    if img and not (isinstance(img, str) and img.startswith("data:image/") and len(img) < 5_000_000):
        abort(400, "Invalid or oversized image.")
    if not KEY:
        abort(503, "Server is missing OPENROUTER_API_KEY.")
    text = text or "Describe this image."

    cid, conv = d.get("conversation_id"), None
    if cid:
        rows = q("SELECT * FROM conversations WHERE id=? AND user_id=?", (cid, u))
        conv = rows[0] if rows else None
    if not conv:
        cid = q("INSERT INTO conversations(user_id,title,mode,created) VALUES(?,?,?,?)", (u, text[:48], "general", time.time()), w=True)
        conv = {"id": cid, "mode": "general"}

    req = d.get("mode", "auto")
    mode = req if req in MODES else detect(text, conv["mode"])
    hist = q("SELECT role, content FROM messages WHERE conv_id=? ORDER BY id DESC LIMIT 24", (cid,))[::-1]
    content = [{"type": "text", "text": text}, {"type": "image_url", "image_url": {"url": img}}] if img else text
    msgs = [{"role": "system", "content": build_system(u, mode)}] + hist + [{"role": "user", "content": content}]

    try:
        raw = llm(msgs)
    except (requests.RequestException, KeyError, IndexError) as e:
        log.error("OpenRouter failure: %s", e)
        abort(502, "The AI service is unavailable right now. Try again shortly.")

    facts = [f.strip()[:200] for f in MEM_TAG.findall(raw) if f.strip()]
    reply = MEM_TAG.sub("", raw).strip() or "I have nothing to add to that."
    if not facts and FACT.match(text) and len(text) < 200:
        facts = [text]
    now = time.time()
    for f in facts:
        q("INSERT OR IGNORE INTO memories(user_id,kind,content,created) VALUES(?,?,?,?)", (u, "auto", f, now), w=True)
    q("INSERT INTO messages(conv_id,role,content,created) VALUES(?,?,?,?)", (cid, "user", text + (" [image]" if img else ""), now), w=True)
    q("INSERT INTO messages(conv_id,role,content,created) VALUES(?,?,?,?)", (cid, "assistant", reply, now), w=True)
    q("UPDATE conversations SET mode=? WHERE id=?", (mode, cid), w=True)
    log.info("chat user=%s mode=%s facts=%d", u[:8], mode, len(facts))
    return jsonify(reply=reply, mode=mode, conversation_id=cid, memories=facts)


# ---------- conversations ----------
@app.get("/api/conversations")
def conversations():
    return jsonify(q("SELECT id,title,mode,created FROM conversations WHERE user_id=? ORDER BY id DESC LIMIT 100", (user(),)))


@app.get("/api/conversations/<int:cid>")
def conversation(cid):
    if not q("SELECT 1 FROM conversations WHERE id=? AND user_id=?", (cid, user())):
        abort(404, "Conversation not found.")
    return jsonify(q("SELECT role, content FROM messages WHERE conv_id=? ORDER BY id", (cid,)))


@app.delete("/api/conversations/<int:cid>")
def delete_conversation(cid):
    u = user()
    q("DELETE FROM messages WHERE conv_id IN (SELECT id FROM conversations WHERE id=? AND user_id=?)", (cid, u), w=True)
    q("DELETE FROM conversations WHERE id=? AND user_id=?", (cid, u), w=True)
    return jsonify(ok=True)


# ---------- settings ----------
@app.route("/api/settings", methods=["GET", "PUT"])
def settings():
    u = user()
    if request.method == "PUT":
        for k, v in (request.get_json(silent=True) or {}).items():
            if isinstance(k, str) and 0 < len(k) < 40:
                q("INSERT OR REPLACE INTO settings(user_id,key,value) VALUES(?,?,?)", (u, k, str(v)[:500]), w=True)
    return jsonify({r["key"]: r["value"] for r in q("SELECT key, value FROM settings WHERE user_id=?", (u,))})


# ---------- memories / projects / notes / saved responses ----------
RES = {
    "memories": ("memories", ["content"]),
    "projects": ("projects", ["name", "description"]),
    "notes": ("notes", ["title", "body"]),
    "saved": ("saved_responses", ["content"]),
}


@app.route("/api/r/<res>", methods=["GET", "POST"])
def resource(res):
    if res not in RES:
        abort(404, "Unknown resource.")
    u = user()
    table, fields = RES[res]  # trusted constants, never user input
    if request.method == "POST":
        d = request.get_json(silent=True) or {}
        vals = [str(d.get(f, "")).strip()[:5000] for f in fields]
        if not vals[0]:
            abort(400, f"Missing {fields[0]}.")
        q(f"INSERT OR IGNORE INTO {table}(user_id,{','.join(fields)},created) VALUES(?{',?' * len(fields)},?)", (u, *vals, time.time()), w=True)
    return jsonify(q(f"SELECT * FROM {table} WHERE user_id=? ORDER BY id DESC LIMIT 200", (u,)))


@app.delete("/api/r/<res>/<int:rid>")
def resource_delete(res, rid):
    if res not in RES:
        abort(404, "Unknown resource.")
    q(f"DELETE FROM {RES[res][0]} WHERE id=? AND user_id=?", (rid, user()), w=True)
    return jsonify(ok=True)


@app.get("/api/health")
def health():
    return jsonify(ok=True, model=MODEL, configured=bool(KEY))


if __name__ == "__main__":
    app.run(port=int(os.environ.get("PORT", 5000)))
