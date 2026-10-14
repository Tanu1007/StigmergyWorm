"""
Webcam frame server. Run with Windows Python (WSL can't access the webcam):

    py -m pip install imageio imageio-ffmpeg pillow psutil
    py win_camera_server.py

Endpoints:
    GET /health  -> {"ok": true, "frames": N}
    GET /frame   -> image/jpeg (most recent frame)
"""

import io
import json
import os
import re
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import imageio_ffmpeg
from PIL import Image, ImageStat

PORT = 8765
CAMERA_INDEX = 0        # 0 = first video device; 1 for a second camera
DEVICE_NAME = None      # or an exact device name, e.g. "Integrated Camera"
MAX_W = 768             # downscale so the model gets a small image
QUALITY = 70            # JPEG quality
WARMUP_S = 1.5          # webcams send black frames while exposure settles
BLACK_MEAN = 8          # mean brightness (0-255) at or below this = black frame

lock = threading.Lock()
latest_jpeg: bytes | None = None
frame_count = 0
stop = threading.Event()


def find_camera() -> str:
    """Return the DirectShow name of the camera to open."""
    if DEVICE_NAME:
        return DEVICE_NAME
    proc = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner",
         "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
        capture_output=True, encoding="utf-8", errors="replace",
    )
    # ffmpeg lists devices on stderr as lines like:  [dshow @ ...] "Name" (video)
    cams = re.findall(r'^\[dshow[^\]]*\]\s+"(.+)" \(video\)', proc.stderr, re.M)
    if CAMERA_INDEX >= len(cams):
        raise RuntimeError(f"camera index {CAMERA_INDEX} not found; cameras: {cams}")
    return cams[CAMERA_INDEX]


def capture_loop():
    """Keep `latest_jpeg` updated with the newest usable frame until `stop`."""
    global latest_jpeg, frame_count
    while not stop.is_set():
        frames = None
        try:
            name = find_camera()
            frames = imageio_ffmpeg.read_frames(
                f"video={name}", input_params=["-f", "dshow"], pix_fmt="rgb24")
            size = next(frames)["size"]     # first item is metadata
            print(f"[camera] opened {name!r} @ {size[0]}x{size[1]}", flush=True)
            opened = time.time()
            warned = False

            for raw in frames:
                if stop.is_set():
                    break
                if time.time() - opened < WARMUP_S:
                    continue
                img = Image.frombytes("RGB", size, raw)
                # Never publish black frames; the model can't use them.
                if max(ImageStat.Stat(img).mean) <= BLACK_MEAN:
                    if not warned:
                        print("[camera] frames are black: lens covered, privacy "
                              "shutter closed, or camera in use elsewhere?", flush=True)
                        warned = True
                    continue
                warned = False

                if img.width > MAX_W:
                    img = img.resize((MAX_W, round(img.height * MAX_W / img.width)),
                                     Image.LANCZOS)
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=QUALITY)
                with lock:
                    latest_jpeg = buf.getvalue()
                    frame_count += 1
        except Exception as e:
            if not stop.is_set():
                print(f"[camera] error: {e!r}, retrying in 2s", flush=True)
                stop.wait(2)
        finally:
            # Closing the generator stops ffmpeg, which releases the webcam.
            # ffmpeg ignores Ctrl+C, so without this it keeps the camera on.
            if frames is not None:
                frames.close()


def kill_child_processes():
    """Last resort if ffmpeg is still running after shutdown."""
    try:
        import psutil
    except ImportError:
        return
    for child in psutil.Process(os.getpid()).children(recursive=True):
        child.kill()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass    # keep the console readable

    def send(self, ctype, body, code=200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        with lock:
            jpeg, count = latest_jpeg, frame_count
        if self.path.startswith("/health"):
            self.send("application/json", json.dumps({"ok": True, "frames": count}).encode())
        elif self.path.startswith("/frame"):
            if jpeg is None:
                self.send("text/plain", b"no frame captured yet", 503)
            else:
                self.send("image/jpeg", jpeg)
        else:
            self.send("text/plain", b"try /frame or /health", 404)


if __name__ == "__main__":
    capture = threading.Thread(target=capture_loop, daemon=True)
    capture.start()
    # Bind 0.0.0.0 (not localhost) so WSL can reach it too.
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"[server] http://0.0.0.0:{PORT}  (/frame, /health)  Ctrl+C to stop",
          flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[server] stopping...", flush=True)
    finally:
        stop.set()
        server.server_close()
        capture.join(timeout=3)
        if capture.is_alive():
            kill_child_processes()
        print("[server] camera released", flush=True)
