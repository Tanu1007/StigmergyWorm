"""Checks every link in the chain and says exactly what to fix. Run this first."""

import sys

import requests

import config


def ok(msg):
    print(f"  \033[32m[ok]\033[0m {msg}")


def bad(msg, fix):
    print(f"  \033[31m[--]\033[0m {msg}\n       fix: {fix}")


def main():
    print(f"\nWSL detected: {config.is_wsl()}")
    if config.is_wsl():
        print(f"Windows host IP: {config.windows_host_ip()}")

    print("\n1. Ollama reachable")
    try:
        base = config.ollama()
        ok(f"{base}")
    except RuntimeError as e:
        bad("cannot reach Ollama", "")
        print(f"\n{e}\n")
        print("If Ollama runs on Windows, it binds 127.0.0.1 by default and")
        print("WSL cannot see it. In a Windows PowerShell, set it permanently:")
        print("  [Environment]::SetEnvironmentVariable("
              "'OLLAMA_HOST','0.0.0.0',[EnvironmentVariableTarget]::User)")
        print("  [Environment]::SetEnvironmentVariable("
              "'OLLAMA_ORIGINS','*',[EnvironmentVariableTarget]::User)")
        print("then quit Ollama from the system tray and relaunch it.")
        sys.exit(1)

    print("\n2. Vision models installed")
    tags = requests.get(base + "/api/tags", timeout=10).json()
    names = [m["name"] for m in tags.get("models", [])]
    for n in names:
        print(f"       {n}")
    vision = [n for n in names if "vl" in n or "gemma4" in n or "llava" in n]
    if vision:
        ok(f"{len(vision)} vision-capable model(s)")
    else:
        bad("no vision model found", "ollama pull qwen3-vl:8b-instruct")

    print("\n3. Camera server")
    try:
        cam = config.camera()
        n = requests.get(cam + "/health", timeout=5).json().get("frames", 0)
        if n > 0:
            ok(f"{cam} — {n} frames captured")
        else:
            bad(f"{cam} is up but has captured 0 frames",
                "the webcam is busy or the wrong camera is selected; "
                "close Teams/Zoom, or set CAMERA_INDEX = 1 in win_camera_server.py")
    except RuntimeError:
        bad("camera server not reachable",
            "on WINDOWS run:  py win_camera_server.py   "
            "(WSL cannot access the webcam directly)")

    print("\n4. End-to-end inference")
    if vision:
        model = next((v for v in vision if "vl" in v), vision[0])
        try:
            r = requests.post(base + "/api/chat", timeout=300, json={
                "model": model, "stream": False,
                "messages": [{"role": "user", "content": "Reply with just: READY"}],
            })
            ok(f"{model} -> {r.json()['message']['content'].strip()[:40]}")
        except Exception as e:
            bad(f"inference failed: {e!r}", "check `ollama serve` logs")

    print()


if __name__ == "__main__":
    main()
