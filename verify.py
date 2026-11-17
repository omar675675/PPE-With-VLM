"""
VLM check of the people the PPE model flagged.

For each flagged person, the person's crop is sent to a vision-language model
(through Ollama) with a plain question: "is this person wearing X?". If the
VLM says the item is missing, that violation is verified. If it says the item
is there, the violation is marked for a human to decide: accept the VLM's
answer (no violation) or keep the violation.

Each violation type of a person is checked and decided on its own, so a person
flagged for both no_helmet and no_vest can be verified for one and not the other.

Results go to <output_dir>/review.json, keyed by "<video>/<track_id>", then by type:
    {"ppe3/22": {"no_vest": {"state": "verified", "answer": false, "reason": "..."}}}
state is one of: pending, verified, rejected, error,
    kept (human: the violation is real) or accepted (human: the VLM is right, no violation)
"""

import base64
import json
import os
import re
import threading
import urllib.request

PROMPT = """This is a crop of one worker on a construction site.
Judge ONLY the main person in the centre of the image; ignore anyone cut off at the edges.
Is that person wearing:
{items}
Reply only with JSON: {fmt}"""

DONE = ("verified", "rejected", "kept", "accepted", "deleted")  # not checked again ("deleted" = old name of accepted)


def key(record):
    return f"{record['video']}/{record['track_id']}"


class Reviewer:
    def __init__(self, cfg, out_dir):
        self.cfg = cfg
        self.vc = cfg.get("vlm", {})
        self.checks = cfg["ppe"]["checks"]
        self.path = out_dir / "review.json"
        self.out_dir = out_dir
        self.lock = threading.Lock()
        self.queue = []       # (record, [types]) waiting for the VLM
        self.progress = None  # {"done": n, "total": m} while the worker is running

        # "pending" left over from a previous dashboard run will never finish - forget it
        with self.lock:
            data = self.load()
            for entry in data.values():
                for t in [t for t, r in entry.items() if r.get("state") == "pending"]:
                    del entry[t]
            if data:
                self._save(data)

    # --- review.json ---------------------------------------------------------

    def load(self):
        try:
            return json.loads(self.path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save(self, data):
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2))
        os.replace(tmp, self.path)

    def _update(self, k, t, **fields):
        """Set fields on one type of one person. Call with self.lock held."""
        data = self.load()
        entry = data.setdefault(k, {})
        entry[t] = {**entry.get(t, {}), **fields}
        self._save(data)

    # --- actions -------------------------------------------------------------

    def start(self, jobs):
        """Queue (record, [types]) jobs. Types already answered are skipped. Returns how many were queued."""
        with self.lock:
            data = self.load()
            queued = 0
            for record, types in jobs:
                k = key(record)
                types = [t for t in types if t in record["types"]
                         and data.get(k, {}).get(t, {}).get("state") not in DONE + ("pending",)]
                if not types:
                    continue
                for t in types:
                    data.setdefault(k, {})[t] = {"state": "pending"}
                self.queue.append((record, types))
                queued += 1
            if not queued:
                return 0
            self._save(data)
            if self.progress is None:
                self.progress = {"done": 0, "total": queued}
                threading.Thread(target=self._run, daemon=True).start()
            else:
                self.progress["total"] += queued
        return queued

    def decide(self, k, t, action):
        """Human decision on one violation of one person: 'keep' (violation is real) or 'accept' (VLM is right)."""
        with self.lock:
            self._update(k, t, state={"keep": "kept", "accept": "accepted"}[action])

    def _run(self):
        while True:
            with self.lock:
                if not self.queue:
                    self.progress = None  # under the lock, so start() knows to begin a new worker
                    break
                record, types = self.queue.pop(0)
            try:
                answer, reason = self.ask(record, types)
                result = {t: {"state": "rejected" if answer[t] else "verified",  # sees the item -> disagrees
                              "answer": answer[t], "reason": reason} for t in types}
            except Exception as e:
                result = {t: {"state": "error", "reason": str(e)} for t in types}
            with self.lock:
                for t, fields in result.items():
                    self._update(key(record), t, **fields)
                self.progress["done"] += 1
        self.unload()

    # --- the VLM -------------------------------------------------------------

    def _post(self, path, body, timeout):
        req = urllib.request.Request(self.vc.get("url", "http://localhost:11434").rstrip("/") + path,
                                     json.dumps(body).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.load(resp)

    def ask(self, record, types):
        """Ask the VLM about these violation types. Returns ({type: is_wearing_the_item}, reason)."""
        items = {t: self.checks[t]["good"] for t in types}  # no_vest -> vest
        desc = self.vc.get("items", {})
        prompt = PROMPT.format(
            items="\n".join(f"- {i}: {desc.get(i, i)}" for i in items.values()),
            fmt="{" + ", ".join(f'"{i}": true or false' for i in items.values()) + ', "reason": "short reason"}',
        )
        img = base64.b64encode((self.out_dir / record["crop"]).read_bytes()).decode()
        text = self._post("/api/chat", {
            "model": self.vc.get("model", "qwen3-vl:8b"),
            "stream": False,
            "think": False,
            "options": {"temperature": 0},
            "messages": [{"role": "user", "content": prompt, "images": [img]}],
        }, self.vc.get("timeout_s", 120))["message"]["content"]

        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise ValueError(f"no JSON in VLM reply: {text[:200]!r}")
        raw = json.loads(m.group(0))
        answer = {}
        for t, i in items.items():
            v = raw.get(i)
            if isinstance(v, str):
                v = v.strip().lower() in ("true", "yes")
            if not isinstance(v, bool):
                raise ValueError(f"VLM reply has no '{i}': {text[:200]!r}")
            answer[t] = v
        return answer, str(raw.get("reason", "")).strip()

    def unload(self):
        """Free the VLM's GPU memory, so the PPE models have room on the next run."""
        try:
            self._post("/api/generate", {"model": self.vc.get("model", "qwen3-vl:8b"), "keep_alive": 0}, 30)
        except Exception:
            pass  # not critical: Ollama unloads it by itself after a few minutes
