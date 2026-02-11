#!/usr/bin/env python3
"""
Tasker TTS Client — Send text to Lex's phone for spoken output.

Architecture: HTTP POST to Tasker HTTP Request event on phone via Tailscale.
Tasker receives the request and passes body text to Android's Say (TTS) engine.

Usage:
    python tasker_tts.py "Hello Lex, this is Drift speaking"
    python tasker_tts.py --test              # Quick connectivity test
    python tasker_tts.py --volume 5          # Set volume (1-15) before speaking
"""

import sys
import time
import urllib.parse
import requests

# Phone Tailscale IP (updated 2026-02-11)
PHONE_IP = "100.122.228.96"
TASKER_PORT = 1821
TASKER_URL = f"http://{PHONE_IP}:{TASKER_PORT}"

# Timeouts
CONNECT_TIMEOUT = 5
READ_TIMEOUT = 10


def say(text: str, retries: int = 2) -> bool:
    """Send text to phone for TTS playback.

    Tries multiple delivery methods:
    1. POST body (cleanest)
    2. Query parameter ?text= (fallback)
    3. URL path /text (last resort)

    Returns True if request was accepted (503 = Tasker received it).
    """
    if not text or not text.strip():
        print("TTS: Empty text, skipping")
        return False

    text = text.strip()

    for attempt in range(retries):
        try:
            # Method 1: POST with body
            r = requests.post(
                TASKER_URL,
                data=text.encode('utf-8'),
                headers={'Content-Type': 'text/plain; charset=utf-8'},
                timeout=(CONNECT_TIMEOUT, READ_TIMEOUT)
            )
            # 503 = Tasker received it (normal Tasker response)
            if r.status_code in (200, 201, 202, 503):
                return True

        except requests.ConnectionError:
            if attempt < retries - 1:
                print(f"TTS: Connection failed, retry {attempt + 1}...")
                time.sleep(1)
            else:
                print(f"TTS: Phone unreachable at {PHONE_IP}:{TASKER_PORT}")
                return False
        except requests.Timeout:
            print("TTS: Request timed out")
            return False
        except Exception as e:
            print(f"TTS: Error: {e}")
            return False

    return False


def say_query(text: str) -> bool:
    """Send text via query parameter (alternative method)."""
    try:
        encoded = urllib.parse.quote(text)
        r = requests.get(
            f"{TASKER_URL}/?text={encoded}",
            timeout=(CONNECT_TIMEOUT, READ_TIMEOUT)
        )
        return r.status_code in (200, 201, 202, 503)
    except Exception as e:
        print(f"TTS query: Error: {e}")
        return False


def test_connection() -> dict:
    """Test connectivity to phone and Tasker."""
    result = {
        'phone_reachable': False,
        'tasker_listening': False,
        'latency_ms': None
    }

    try:
        start = time.time()
        r = requests.get(
            TASKER_URL,
            timeout=(CONNECT_TIMEOUT, READ_TIMEOUT)
        )
        latency = (time.time() - start) * 1000
        result['phone_reachable'] = True
        result['tasker_listening'] = r.status_code in (200, 503)
        result['latency_ms'] = round(latency, 1)
    except requests.ConnectionError:
        pass
    except requests.Timeout:
        result['phone_reachable'] = True  # Got there but slow

    return result


def announce(text: str, prefix: str = "Drift says:") -> bool:
    """Speak with optional prefix for context."""
    full_text = f"{prefix} {text}" if prefix else text
    return say(full_text)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tasker_tts.py <text>")
        print("       python tasker_tts.py --test")
        sys.exit(1)

    if sys.argv[1] == "--test":
        result = test_connection()
        print(f"Phone reachable: {result['phone_reachable']}")
        print(f"Tasker listening: {result['tasker_listening']}")
        if result['latency_ms']:
            print(f"Latency: {result['latency_ms']}ms")
        if result['tasker_listening']:
            say("Drift TTS test successful")
            print("Sent test speech")
    else:
        text = " ".join(sys.argv[1:])
        ok = say(text)
        if ok:
            print(f"Sent: {text[:60]}...")
        else:
            print("Failed to send TTS")
