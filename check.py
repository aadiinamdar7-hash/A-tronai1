"""Smoke test for the MARZ backend. Run with: python check.py"""
# flake8: noqa: E501
import io
import os
import sys
import tempfile

os.environ["DB_PATH"] = os.path.join(tempfile.mkdtemp(), "check.db")
os.environ["OPENROUTER_API_KEY"] = "test-key"

import app as marz  # noqa: E402

FAILED = []
HEAD = {"X-User-ID": "check-user-0123456789"}
IMAGE = "data:image/jpeg;base64,AAAA"
FACT = "User studies GCSE Computer Science"


def check(name, ok):
    print(("PASS  " if ok else "FAIL  ") + name)
    if not ok:
        FAILED.append(name)


def fake_model(messages):
    return "Hello! [[MEMORY: " + FACT + "]]"


marz.call_openrouter = fake_model
client = marz.app.test_client()


def post_chat(**body):
    return client.post("/api/chat", json=body, headers=HEAD)


def upload(size, mime="image/png"):
    picture = (io.BytesIO(b"x" * size), "a.png", mime)
    return client.post(
        "/api/chat",
        data={"message": "what is this", "image": picture},
        headers=HEAD,
        content_type="multipart/form-data",
    )


def get_json(path):
    return client.get(path, headers=HEAD).json


health = client.get("/api/health")
check("health endpoint", health.status_code == 200 and health.json["ok"])
check("missing user id rejected", client.get("/api/conversations").status_code == 401)

first = post_chat(message="I am studying GCSE Computer Science")
cid = first.json.get("conversation_id")
check("chat returns a clean reply", first.status_code == 200 and "[[" not in first.json["reply"])
check("memory marker stored", first.json["memories"] == [FACT])
check("conversation listed", len(get_json("/api/conversations")) == 1)
check("history has 2 messages", len(get_json("/api/conversations/%d" % cid)) == 2)
check("long-term memory saved", [m["content"] for m in get_json("/api/r/memories")] == [FACT])

SAMPLES = (
    ("notes", {"title": "t", "body": "b"}),
    ("projects", {"name": "p", "description": "d"}),
    ("saved", {"content": "c"}),
)
for res, body in SAMPLES:
    client.post("/api/r/" + res, json=body, headers=HEAD)
    rows = get_json("/api/r/" + res)
    check(res + " create and list", len(rows) == 1)
    done = client.delete("/api/r/%s/%d" % (res, rows[0]["id"]), headers=HEAD)
    check(res + " delete", done.status_code == 200 and get_json("/api/r/" + res) == [])

saved = client.put("/api/profile", json={"name": "Sam", "bio": "Student"}, headers=HEAD)
check("profile saved", saved.json["name"] == "Sam")
prefs = client.put("/api/settings", json={"theme": "dark"}, headers=HEAD)
check("settings saved", prefs.json["theme"] == "dark")

check("JSON image accepted", post_chat(message="hi", image=IMAGE).status_code == 200)
check("bad image rejected", post_chat(message="hi", image="data:text/html;base64,AAAA").status_code == 400)
check("file upload accepted", upload(10).status_code == 200)
check("oversized upload rejected", upload(4 * 1024 * 1024 + 1).status_code == 400)
check("wrong upload type rejected", upload(10, "text/plain").status_code == 400)

client.delete("/api/conversations/%d" % cid, headers=HEAD)
check("conversation deleted", client.get("/api/conversations/%d" % cid, headers=HEAD).status_code == 404)
check("unknown resource 404", client.get("/api/r/nope", headers=HEAD).status_code == 404)

if FAILED:
    print("\n%d check(s) failed." % len(FAILED))
    sys.exit(1)
print("\nALL CHECKS PASSED")
