"""Explicit local-only Ollama wire adapter; images are encoded only in transit."""
import base64
import hashlib
import ipaddress
import json
from pathlib import Path
from urllib.parse import urlsplit
import requests


def local_url(value):
    p = urlsplit(value)
    if p.scheme not in ("http", "https") or not p.hostname or p.username or p.password:
        raise ValueError("Use an explicit local Ollama HTTP endpoint without credentials")
    if p.query or p.fragment or p.path not in ("", "/"):
        raise ValueError("Ollama URL must be an origin")
    try:
        ip = ipaddress.ip_address(p.hostname)
        local = ip.is_loopback or (ip.version == 4 and any(ip in net for net in
            (ipaddress.ip_network("10.0.0.0/8"), ipaddress.ip_network("172.16.0.0/12"),
             ipaddress.ip_network("192.168.0.0/16"))))
    except ValueError:
        local = p.hostname == "localhost"
    if not local:
        raise ValueError("Only localhost or explicit RFC1918 addresses are permitted")
    _ = p.port  # reject malformed ports
    return value.rstrip("/")


class Frame:
    def __init__(self, path, source="supplied_frame", description=None):
        self.path = Path(path)
        if not isinstance(source, str) or not 1 <= len(source) <= 80:
            raise ValueError("Frame source label must be 1-80 characters")
        if description is not None and (not isinstance(description, str) or len(description) > 400):
            raise ValueError("Frame description must be at most 400 characters")
        self.source, self.description = source, description
        raw = self.path.read_bytes()
        if not raw or len(raw) > 8 * 1024 * 1024:
            raise ValueError("Image must be nonempty and at most 8 MiB")
        if not (raw.startswith(b"\x89PNG\r\n\x1a\n") or raw.startswith(b"\xff\xd8\xff")):
            raise ValueError("Only PNG or JPEG evidence frames are accepted")
        self._raw = raw
        self.sha256 = hashlib.sha256(raw).hexdigest()

    def metadata(self):
        meta = {"sha256": self.sha256, "bytes": len(self._raw), "name": self.path.name,
                "source": self.source, "trust": "untrusted_observation"}
        if self.description:
            meta["description"] = self.description
        return meta

    def wire(self):
        return base64.b64encode(self._raw).decode("ascii")


class OllamaModel:
    def __init__(self, model="qwen3-vl:8b-instruct", base_url="http://127.0.0.1:11434", timeout=120):
        self.model = model
        self.base_url = local_url(base_url)
        if not 1 <= timeout <= 300:
            raise ValueError("Timeout must be 1–300 seconds")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.trust_env = False
        self.model_digest = None

    def preflight(self):
        r = self.session.get(self.base_url + "/api/tags", timeout=15, allow_redirects=False)
        r.raise_for_status()
        candidates = [x for x in r.json().get("models", []) if x.get("name") == self.model or x.get("model") == self.model]
        if len(candidates) != 1:
            raise ValueError("The exact requested model tag must already be installed locally")
        self.model_digest = candidates[0].get("digest")
        if not self.model_digest:
            raise ValueError("Ollama did not provide the model digest")
        return {"model": self.model, "digest": self.model_digest, "endpoint": self.base_url}

    def complete(self, role, system, payload, schema, frame=None):
        if self.model_digest is None:
            raise RuntimeError("Call preflight before inference")
        message = {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}
        if frame is not None:
            message["images"] = [frame.wire()]
        r = self.session.post(self.base_url + "/api/chat", json={
            "model": self.model, "messages": [{"role": "system", "content": system}, message],
            "format": schema, "stream": False, "think": False,
            "options": {"temperature": 0, "num_ctx": 8192, "num_predict": 512}},
            timeout=self.timeout, allow_redirects=False)
        if 300 <= r.status_code < 400:
            raise RuntimeError("Local model redirects are not followed")
        r.raise_for_status()
        body = r.json()
        if body.get("done_reason") == "length":
            raise ValueError("Model output was truncated; this attempt is not a valid decision")
        content = body.get("message", {}).get("content")
        if not isinstance(content, str):
            raise ValueError("Missing model response content")
        return json.loads(content)
