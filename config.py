"""Finds Ollama and the camera server: on this machine, or on the Windows host from WSL."""

import os
import re

import requests

OLLAMA_PORT = 11434
CAMERA_PORT = 8765


def is_wsl() -> bool:
    try:
        with open("/proc/version") as f:
            return "microsoft" in f.read().lower()
    except FileNotFoundError:
        return False


def windows_host_ip() -> str | None:
    """The Windows host's IP as seen from WSL: the nameserver in resolv.conf."""
    try:
        with open("/etc/resolv.conf") as f:
            m = re.search(r"^nameserver\s+([\d.]+)", f.read(), re.M)
            return m.group(1) if m else None
    except FileNotFoundError:
        return None


def resolve(service: str, port: int, probe_path: str) -> str:
    """Return the first base URL that answers: $<SERVICE>_HOST, localhost, Windows host."""
    override = os.environ.get(f"{service.upper()}_HOST")
    hosts = ([override] if override else []) + ["127.0.0.1"]
    if is_wsl() and (ip := windows_host_ip()):
        hosts.append(ip)

    for host in hosts:
        base = host if host.startswith("http") else f"http://{host}:{port}"
        try:
            if requests.get(base + probe_path, timeout=1.5).ok:
                return base
        except requests.RequestException:
            pass

    raise RuntimeError(
        f"Could not reach {service} on any of: {hosts}\n"
        f"  - Is it running?\n"
        f"  - If it's on Windows and you're in WSL, it must bind 0.0.0.0.\n"
        f"  - Check the Windows firewall allows inbound on port {port}.\n"
        f"  - Or set {service.upper()}_HOST explicitly in .env"
    )


_urls: dict[str, str] = {}   # resolved once, on first use


def ollama() -> str:
    if "ollama" not in _urls:
        _urls["ollama"] = resolve("ollama", OLLAMA_PORT, "/api/tags")
    return _urls["ollama"]


def camera() -> str:
    if "camera" not in _urls:
        _urls["camera"] = resolve("camera", CAMERA_PORT, "/health")
    return _urls["camera"]
