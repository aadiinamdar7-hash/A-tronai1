"""MARZ backend: Flask, SQLite and OpenRouter.

Local run: python app.py
Production run: gunicorn app:app --workers 1 --threads 4 --timeout 90
"""

import base64
import logging
import os
import re
import sqlite3
import time
from contextlib import closing

import requests
from dotenv import load_dotenv
from flask import Flask, abort, g, jsonify, request
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.exceptions import HTTPException
from werkzeug.middleware.proxy_fix import ProxyFix

load_dotenv()

HTTPS = "https" + "://"
HTTP = "http" + "://"

OPENROUTER_URL = HTTPS + "openrouter.ai/api/v1/chat/completions"

KEY = os.environ.get("OPENROUTER_API_KEY")
MODEL = os.environ.get("MARZ_MODEL", "openrouter/auto")
DB_PATH = os.environ.get("DB_PATH", "marz.db")
SITE_URL = os.environ.get("SITE_URL", HTTP + "localhost:8080")
RAW_ORIGINS = os.environ.get("ALLOWED_ORIGINS", HTTP + "localhost:8080")
ORIGINS = [origin.strip() for origin in RAW_ORIGINS.split(",") if origin.strip()]

MAX_IMAGE_BYTES = 4 * 1024 * 1024
MAX_DATA_URL_CHARS = 5_000_000

IMAGE_TYPES = (
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/gif",
)

DATA_URL_RE = re.compile(
    r"data:image/(?:png|jpeg|webp|gif);base64,[A-Za-z0-9+/=]+"
)

USER_ID_RE = re.compile(r"[A-Za-z0-9-]{16,64}")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

log = logging.getLogger("marz")


def var(spec):
    """Build a Flask URL variable from its specification."""
    return chr(60) + spec + chr(62)


app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1)

app.config["MAX_CONTENT_LENGTH"] = 6 * 1024 * 1024

CORS(
    app,
    origins=ORIGINS,
    allow_headers=["Content-Type", "X-User-ID"],
)

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["300/hour"],
    storage_uri="memory://",
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS users(
    id TEXT PRIMARY KEY,
    created REAL
);

CREATE TABLE IF NOT EXISTS memories(
    id INTEGER PRIMARY KEY,
    user_id TEXT,
    kind TEXT DEFAULT 'manual',
    content TEXT,
    created REAL,
    UNIQUE(user_id, content)
);

CREATE TABLE IF NOT EXISTS conversations(
    id INTEGER PRIMARY KEY,
    user_id TEXT,
    title TEXT,
    mode TEXT,
    created REAL
);

CREATE TABLE IF NOT EXISTS messages(
    id INTEGER PRIMARY KEY,
    conv_id INTEGER,
    role TEXT,
    content TEXT,
    created REAL
);

CREATE TABLE IF NOT EXISTS projects(
    id INTEGER PRIMARY KEY,
    user_id TEXT,
    name TEXT,
    description TEXT,
    created REAL
);

CREATE TABLE IF NOT EXISTS settings(
    user_id TEXT,
    key TEXT,
    value TEXT,
    PRIMARY KEY(user_id, key)
);

CREATE TABLE IF NOT EXISTS saved_responses(
    id INTEGER PRIMARY KEY,
    user_id TEXT,
    content TEXT,
    created REAL
);

CREATE TABLE IF NOT EXISTS notes(
    id INTEGER PRIMARY KEY,
    user_id TEXT,
    title TEXT,
    body TEXT,
    created REAL
);

CREATE INDEX IF NOT EXISTS idx_messages_conv
ON messages(conv_id);

CREATE INDEX IF NOT EXISTS idx_memories_user
ON memories(user_id);
"""


def init_db():
    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(SCHEMA)


init_db()


def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH, timeout=10)
        g.db.row_factory = sqlite3.Row

    return g.db


@app.teardown_appcontext
def close_db(exc):
    conn = g.pop("db", None)

    if conn is not None:
        conn.close()


def query(sql, args=()):
    cursor = get_db().execute(sql, args)
    return [dict(row) for row in cursor.fetchall()]


def execute(sql, args=()):
    conn = get_db()
    cursor = conn.execute(sql, args)
    conn.commit()
    return cursor.lastrowid


def current_user():
    user_id = request.headers.get("X-User-ID", "")

    if not USER_ID_RE.fullmatch(user_id):
        abort(401, "Missing or invalid X-User-ID.")

    execute(
        "INSERT OR IGNORE INTO users(id, created) VALUES(?, ?)",
        (user_id, time.time()),
    )

    return user_id


def to_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


@app.errorhandler(Exception)
def handle_error(err):
    if isinstance(err, HTTPException):
        return jsonify(error=err.description), err.code

    log.exception("Unhandled error")
    return jsonify(error="Something went wrong on the server."), 500


# Settings and profile

def get_setting(user_id, key, default=""):
    rows = query(
        "SELECT value FROM settings WHERE user_id=? AND key=?",
        (user_id, key),
    )

    return rows[0]["value"] if rows else default


def save_setting(user_id, key, value):
    execute(
        "INSERT OR REPLACE INTO settings(user_id, key, value) "
        "VALUES(?, ?, ?)",
        (user_id, key, value),
    )


@app.route("/api/settings", methods=["GET", "PUT"])
def settings():
    user_id = current_user()

    if request.method == "PUT":
        data = request.get_json(silent=True) or {}

        for key, value in data.items():
            if isinstance(key, str) and 0 < len(key) < 40:
                save_setting(user_id, key, str(value)[:500])

    rows = query(
        "SELECT key, value FROM settings WHERE user_id=?",
        (user_id,),
    )

    return jsonify({row["key"]: row["value"] for row in rows})


@app.route("/api/profile", methods=["GET", "PUT"])
def profile():
    user_id = current_user()

    if request.method == "PUT":
        data = request.get_json(silent=True) or {}

        for key in ("name", "bio"):
            if key in data:
                save_setting(
                    user_id,
                    key,
                    str(data[key]).strip()[:500],
                )

    created = query(
        "SELECT created FROM users WHERE id=?",
        (user_id,),
    )

    return jsonify(
        id=user_id,
        name=get_setting(user_id, "name"),
        bio=get_setting(user_id, "bio"),
        created=created[0]["created"],
    )


# Modes, prompts and long-term memory

BASE_PROMPT = (
    "You are MARZ, a capable, warm and concise personal AI assistant "
    "with a calm, JARVIS-like poise. Use British English. Be accurate "
    "and say when you are unsure."
)

MODES = {
    "general": "Answer helpfully and conversationally.",
    "study": (
        "Act as a patient tutor: explain simply, check understanding, "
        "and offer summaries, flashcards and exam-style questions."
    ),
    "code": (
        "Act as a senior engineer: give working, minimal code in fenced "
        "blocks, and explain bugs, architecture and trade-offs."
    ),
    "research": (
        "Act as a research analyst: give structured, balanced answers, "
        "separate fact from uncertainty, and flag what to verify. "
        "You cannot browse the web."
    ),
    "creative": (
        "Act as a creative writing partner: use a vivid, original voice "
        "and offer variations."
    ),
    "assistant": (
        "Act as a personal assistant: turn requests into plans, "
        "schedules and checklists, and ask at most one question."
    ),
}

HINTS = {
    "code": re.compile(
        r"```|\b(python|javascript|html|css|flask|bug|debug|function|"
        r"stack ?trace|error|api|sql|code)\b",
        re.I,
    ),
    "study": re.compile(
        r"\b(revise|revision|exam|flashcards?|quiz|gcse|a-level|"
        r"homework|study|syllabus)\b",
        re.I,
    ),
    "research": re.compile(
        r"\b(research|compare|versus|pros and cons|how does|why does|"
        r"history of|evidence)\b",
        re.I,
    ),
    "creative": re.compile(
        r"\b(story|poem|lyrics|novel|character|slogan|caption|"
        r"brainstorm)\b",
        re.I,
    ),
    "assistant": re.compile(
        r"\b(remind|schedule|plan my|to-?do|timetable|agenda|"
        r"organi[sz]e)\b",
        re.I,
    ),
}

MEMORY_RE = re.compile(r"\[\[MEMORY:(.*?)\]\]", re.S | re.I)

FACT_RE = re.compile(
    r"^\s*(i am|i'm|i\u2019m|my name is|my goal|i want to|i like|"
    r"i love|i study)\b",
    re.I,
)

MEMORY_RULE = (
    "If the user reveals a durable fact worth remembering, such as "
    "studies, preferences, projects, goals or interests, end your "
    "reply with one final line in exactly this form: "
    "[[MEMORY: short third-person fact]]. Never mention this marker."
)


def detect_mode(text, last):
    for mode, pattern in HINTS.items():
        if pattern.search(text):
            return mode

    return last


def build_system(user_id, mode):
    memories = query(
        "SELECT content FROM memories WHERE user_id=? "
        "ORDER BY created DESC LIMIT 40",
        (user_id,),
    )

    projects = query(
        "SELECT name, description FROM projects WHERE user_id=? "
        "ORDER BY created DESC LIMIT 5",
        (user_id,),
    )

    name = get_setting(user_id, "name")
    bio = get_setting(user_id, "bio")

    parts = [
        BASE_PROMPT,
        "Mode: " + mode + ". " + MODES[mode],
    ]

    if name:
        parts.append("The user's name is " + name + ".")

    if bio:
        parts.append("About the user: " + bio)

    if memories:
        lines = "\n".join(
            "- " + memory["content"]
            for memory in memories
        )

        parts.append(
            "What you know about the user (use naturally, never "
            "recite):\n" + lines
        )

    if projects:
        lines = "\n".join(
            "- " + project["name"] + ": " + project["description"]
            for project in projects
        )

        parts.append("Active projects:\n" + lines)

    parts.append(MEMORY_RULE)

    return "\n".join(parts)


def remember(user_id, fact, kind):
    execute(
        "INSERT OR IGNORE INTO memories(user_id, kind, content, created) "
        "VALUES(?, ?, ?, ?)",
        (user_id, kind, fact, time.time()),
    )


def split_memory(raw, text):
    """Return the user-facing reply and extracted memory facts."""
    facts = [
        fact.strip()[:200]
        for fact in MEMORY_RE.findall(raw)
        if fact.strip()
    ]

    reply = MEMORY_RE.sub("", raw).strip() or "I have nothing to add."

    if not facts and FACT_RE.match(text) and len(text) < 200:
        facts = [text]

    return reply, facts


# OpenRouter

def call_openrouter(messages):
    try:
        response = requests.post(
            OPENROUTER_URL,
            headers={
                "Authorization": "Bearer " + str(KEY),
                "HTTP-Referer": SITE_URL,
                "X-Title": "MARZ",
            },
            json={
                "model": MODEL,
                "messages": messages,
                "max_tokens": 1500,
            },
            timeout=60,
        )

        response.raise_for_status()

        content = response.json()["choices"][0]["message"]["content"]

    except requests.HTTPError as err:
        status = err.response.status_code

        log.error(
            "OpenRouter HTTP %s: %s",
            status,
            err.response.text[:300],
        )

        abort(
            502,
            "OpenRouter returned an error (" + str(status) + ").",
        )

    except (
        requests.RequestException,
        KeyError,
        IndexError,
        ValueError,
    ) as err:
        log.error("OpenRouter failure: %s", err)
        abort(502, "The AI service is unavailable. Try again shortly.")

    return content if isinstance(content, str) else ""


# Chat and image handling

def read_payload():
    if request.mimetype == "multipart/form-data":
        return request.form.to_dict()

    return request.get_json(silent=True) or {}


def read_image(payload):
    """Return an uploaded or JSON image data URL, or None."""
    upload = request.files.get("image")

    if upload is not None:
        if upload.mimetype not in IMAGE_TYPES:
            abort(400, "Unsupported image type.")

        raw = upload.read(MAX_IMAGE_BYTES + 1)

        if not 0 < len(raw) <= MAX_IMAGE_BYTES:
            abort(400, "Image is empty or larger than 4 MB.")

        encoded = base64.b64encode(raw).decode("ascii")

        return "data:" + upload.mimetype + ";base64," + encoded

    data = payload.get("image")

    if not data:
        return None

    valid = (
        isinstance(data, str)
        and len(data) < MAX_DATA_URL_CHARS
        and DATA_URL_RE.fullmatch(data)
    )

    if not valid:
        abort(400, "Invalid or oversized image.")

    try:
        header, encoded = data.split(",", 1)
        decoded = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError):
        abort(400, "Invalid image data.")

    if not decoded or len(decoded) > MAX_IMAGE_BYTES:
        abort(400, "Image is empty or larger than 4 MB.")

    expected_header = header.removeprefix("data:")

    if expected_header not in IMAGE_TYPES:
        abort(400, "Unsupported image type.")

    return data


def user_content(text, image):
    if not image:
        return text

    return [
        {
            "type": "text",
            "text": text,
        },
        {
            "type": "image_url",
            "image_url": {
                "url": image,
            },
        },
    ]


def resolve_conversation(user_id, conversation_id, title):
    if conversation_id is not None:
        rows = query(
            "SELECT id, mode FROM conversations WHERE id=? AND user_id=?",
            (conversation_id, user_id),
        )

        if rows:
            return rows[0]

    new_id = execute(
        "INSERT INTO conversations(user_id, title, mode, created) "
        "VALUES(?, ?, ?, ?)",
        (
            user_id,
            title[:48],
            "general",
            time.time(),
        ),
    )

    return {
        "id": new_id,
        "mode": "general",
    }


def add_message(conversation_id, role, content):
    execute(
        "INSERT INTO messages(conv_id, role, content, created) "
        "VALUES(?, ?, ?, ?)",
        (
            conversation_id,
            role,
            content,
            time.time(),
        ),
    )


@app.post("/api/chat")
@limiter.limit("20/minute")
def chat():
    user_id = current_user()
    payload = read_payload()

    text = str(payload.get("message", "")).strip()[:4000]
    image = read_image(payload)

    if not text and not image:
        abort(400, "Empty message.")

    if not KEY:
        abort(503, "Server is missing OPENROUTER_API_KEY.")

    text = text or "Describe this image."

    conversation = resolve_conversation(
        user_id,
        to_int(payload.get("conversation_id")),
        text,
    )

    conversation_id = conversation["id"]
    requested_mode = str(payload.get("mode", "auto"))

    if requested_mode in MODES:
        mode = requested_mode
    else:
        mode = detect_mode(text, conversation["mode"])

    history = query(
        "SELECT role, content FROM messages WHERE conv_id=? "
        "ORDER BY id DESC LIMIT 24",
        (conversation_id,),
    )[::-1]

    messages = (
        [
            {
                "role": "system",
                "content": build_system(user_id, mode),
            }
        ]
        + history
        + [
            {
                "role": "user",
                "content": user_content(text, image),
            }
        ]
    )

    raw_reply = call_openrouter(messages)
    reply, facts = split_memory(raw_reply, text)

    for fact in facts:
        remember(user_id, fact, "auto")

    add_message(
        conversation_id,
        "user",
        text + (" [image]" if image else ""),
    )

    add_message(conversation_id, "assistant", reply)

    execute(
        "UPDATE conversations SET mode=? WHERE id=?",
        (mode, conversation_id),
    )

    log.info(
        "chat user=%s mode=%s facts=%d",
        user_id[:8],
        mode,
        len(facts),
    )

    return jsonify(
        reply=reply,
        mode=mode,
        conversation_id=conversation_id,
        memories=facts,
    )


# Conversation history

@app.get("/api/conversations")
def conversations():
    rows = query(
        "SELECT id, title, mode, created FROM conversations "
        "WHERE user_id=? ORDER BY id DESC LIMIT 100",
        (current_user(),),
    )

    return jsonify(rows)


@app.get("/api/conversations/" + var("int:cid"))
def conversation(cid):
    user_id = current_user()

    owned = query(
        "SELECT 1 FROM conversations WHERE id=? AND user_id=?",
        (cid, user_id),
    )

    if not owned:
        abort(404, "Conversation not found.")

    rows = query(
        "SELECT role, content FROM messages WHERE conv_id=? ORDER BY id",
        (cid,),
    )

    return jsonify(rows)


@app.delete("/api/conversations/" + var("int:cid"))
def delete_conversation(cid):
    user_id = current_user()

    execute(
        "DELETE FROM messages WHERE conv_id IN "
        "(SELECT id FROM conversations WHERE id=? AND user_id=?)",
        (cid, user_id),
    )

    execute(
        "DELETE FROM conversations WHERE id=? AND user_id=?",
        (cid, user_id),
    )

    return jsonify(ok=True)


# Memories, projects, notes and saved responses

RESOURCES = {
    "memories": ("memories", ("content",)),
    "projects": ("projects", ("name", "description")),
    "notes": ("notes", ("title", "body")),
    "saved": ("saved_responses", ("content",)),
}


@app.route("/api/r/" + var("res"), methods=["GET", "POST"])
def resource(res):
    if res not in RESOURCES:
        abort(404, "Unknown resource.")

    user_id = current_user()
    table, fields = RESOURCES[res]

    if request.method == "POST":
        data = request.get_json(silent=True) or {}

        values = [
            str(data.get(field, "")).strip()[:5000]
            for field in fields
        ]

        if not values[0]:
            abort(400, "Missing " + fields[0] + ".")

        columns = ",".join(fields)
        placeholders = ",".join("?" for _ in fields)

        execute(
            "INSERT OR IGNORE INTO "
            + table
            + "(user_id,"
            + columns
            + ",created) VALUES(?,"
            + placeholders
            + ",?)",
            (user_id, *values, time.time()),
        )

    rows = query(
        "SELECT * FROM "
        + table
        + " WHERE user_id=? ORDER BY id DESC LIMIT 200",
        (user_id,),
    )

    return jsonify(rows)


@app.delete("/api/r/" + var("res") + "/" + var("int:rid"))
def resource_delete(res, rid):
    if res not in RESOURCES:
        abort(404, "Unknown resource.")

    execute(
        "DELETE FROM "
        + RESOURCES[res][0]
        + " WHERE id=? AND user_id=?",
        (rid, current_user()),
    )

    return jsonify(ok=True)


# Health checks

@app.get("/")
def index():
    return jsonify(
        name="MARZ",
        ok=True,
    )


@app.get("/api/health")
def health():
    return jsonify(
        ok=True,
        model=MODEL,
        configured=bool(KEY),
    )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
    )
