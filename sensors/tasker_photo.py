#!/usr/bin/env python3
"""
Tasker Photo Client — Take photos via Tasker HTTP, independent of active app.

Unlike Phone MCP (which requires the MCP server app to be active), this uses
Tasker's HTTP Request event to trigger Android's camera API directly. Works
regardless of which app is in the foreground.

Architecture:
    POST to Tasker HTTP on port 1821 with body starting with "photo:"
    Tasker IF detects "photo:" prefix → runs Take Photo action
    Photo saved to /sdcard/DCIM/Tasker/drift_photo.jpg on phone
    Retrieval: Tasker POSTs the photo back to a local receiver server

Setup required in Tasker (one-time):
    1. Edit existing HTTP Request task (port 1821)
    2. Add IF at top: %http_request_body ~ photo:*
    3. Inside IF: Take Photo (back camera, /DCIM/Tasker/drift_photo.jpg)
    4. Optionally: HTTP Post to http://<laptop_tailscale_ip>:1823/upload
    5. Add ELSE
    6. Move existing Say action into ELSE
    7. End IF

Usage:
    python tasker_photo.py                    # Take photo (back camera)
    python tasker_photo.py front              # Take photo (front camera)
    python tasker_photo.py --receive          # Start receiver server for auto-upload
    python tasker_photo.py --test             # Test connectivity
"""

import sys
import io
import time
import json
import threading
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime, timezone

try:
    import requests
except ImportError:
    requests = None

# Tasker HTTP endpoints
PHONE_IP = "100.122.228.96"
TASKER_TTS_PORT = 1821
TASKER_PHOTO_PORT = 1822  # Separate profile for camera
TASKER_PHOTO_URL = f"http://{PHONE_IP}:{TASKER_PHOTO_PORT}"

# Where received photos are saved locally
SENSOR_DATA_DIR = Path(__file__).parent / "data"
SENSOR_DATA_DIR.mkdir(exist_ok=True)

# Receiver server port (laptop listens, phone pushes photos)
RECEIVER_PORT = 1823

# Timeouts
CONNECT_TIMEOUT = 5
READ_TIMEOUT = 15  # Camera needs more time than TTS


def take_photo(camera: str = "back", retries: int = 2) -> bool:
    """Trigger Tasker to take a photo.

    Sends POST with body 'photo:<camera>' to port 1821.
    Tasker routes this to Take Photo action based on body prefix.

    Args:
        camera: 'back' or 'front'
        retries: Number of retry attempts

    Returns:
        True if Tasker accepted the request (503 = accepted).
    """
    if not requests:
        print("Photo: 'requests' library required")
        return False

    body = f"photo:{camera}"

    for attempt in range(retries):
        try:
            r = requests.post(
                TASKER_PHOTO_URL,
                data=body.encode('utf-8'),
                headers={'Content-Type': 'text/plain; charset=utf-8'},
                timeout=(CONNECT_TIMEOUT, READ_TIMEOUT)
            )
            if r.status_code in (200, 201, 202, 503):
                print(f"Photo: Triggered {camera} camera")
                return True
            else:
                print(f"Photo: Unexpected status {r.status_code}")

        except requests.ConnectionError:
            if attempt < retries - 1:
                print(f"Photo: Connection failed, retry {attempt + 1}...")
                time.sleep(1)
            else:
                print(f"Photo: Phone unreachable at {PHONE_IP}:{TASKER_PORT}")
                return False
        except requests.Timeout:
            # Timeout might be OK — camera takes time, Tasker may not respond quickly
            print("Photo: Request timed out (camera may still have fired)")
            return True  # Assume it worked
        except Exception as e:
            print(f"Photo: Error: {e}")
            return False

    return False


def take_and_announce(camera: str = "back") -> bool:
    """Take photo and announce via TTS."""
    ok = take_photo(camera)
    if ok:
        try:
            from tasker_tts import say
            say(f"Photo taken with {camera} camera")
        except Exception:
            pass
    return ok


class PhotoReceiver(BaseHTTPRequestHandler):
    """HTTP handler that receives photos POSTed by Tasker."""

    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        if content_length == 0:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"No data")
            return

        photo_data = self.rfile.read(content_length)
        ts = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        filepath = SENSOR_DATA_DIR / f"photo_{ts}.jpg"
        filepath.write_bytes(photo_data)

        print(f"Received photo: {filepath} ({len(photo_data):,} bytes)")

        # Auto-index with image embeddings if available
        _auto_index(filepath)

        self.send_response(200)
        self.send_header('Content-Type', 'text/plain')
        self.end_headers()
        self.wfile.write(f"saved:{filepath.name}".encode())

    def log_message(self, format, *args):
        # Suppress default request logging
        pass


def _auto_index(filepath: Path):
    """Try to index photo with jina-clip-v2 embeddings."""
    try:
        sys.path.insert(0, str(filepath.parent))
        from image_search import index_photo, load_index, save_index
        idx = load_index()
        if index_photo(str(filepath), idx):
            save_index(idx)
            print(f"  Indexed: {filepath.name}")
    except Exception as e:
        print(f"  Index skipped: {e}")


def start_receiver(port: int = RECEIVER_PORT, blocking: bool = True):
    """Start HTTP server to receive photos from Tasker.

    Tasker HTTP Post action sends photo to http://<laptop_ip>:<port>/upload
    This server saves it and auto-indexes.

    Args:
        port: Port to listen on (default 1823)
        blocking: If True, blocks forever. If False, returns server + thread.
    """
    server = HTTPServer(('0.0.0.0', port), PhotoReceiver)
    print(f"Photo receiver listening on port {port}")
    print(f"Tasker should POST photos to http://<this_machine>:{port}/")
    print("Press Ctrl+C to stop")

    if blocking:
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nReceiver stopped")
            server.shutdown()
    else:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server, thread


def get_latest_photo() -> Path | None:
    """Get the most recently received photo."""
    photos = sorted(SENSOR_DATA_DIR.glob("photo_*.jpg"), reverse=True)
    return photos[0] if photos else None


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Default: take photo with back camera
        take_photo("back")
    elif sys.argv[1] == "--test":
        from tasker_tts import test_connection
        result = test_connection()
        print(f"Phone reachable: {result['phone_reachable']}")
        print(f"Tasker listening: {result['tasker_listening']}")
        if result['latency_ms']:
            print(f"Latency: {result['latency_ms']}ms")
        if result['tasker_listening']:
            take_photo("back")
            print("Sent photo trigger")
    elif sys.argv[1] == "--receive":
        start_receiver()
    elif sys.argv[1] in ("front", "back"):
        take_photo(sys.argv[1])
    else:
        print("Usage: python tasker_photo.py [front|back|--receive|--test]")
