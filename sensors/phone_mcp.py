#!/usr/bin/env python3
"""
Phone MCP Client — connects Drift to Android phone sensors via MCP.

Uses SSE transport: connect to /sse for session ID, POST JSON-RPC to /message,
read responses from the SSE stream.

Tools available:
    phone_get_cameras   — Camera hardware info
    phone_take_photo    — Take photo (camera_id, quality, flash, focus, zoom, size)
    phone_sensor        — Read all Android sensors

Usage:
    python phone_mcp.py status              # Check connection + list tools
    python phone_mcp.py photo [front|back]  # Take a photo
    python phone_mcp.py sensors             # Read all sensor values
    python phone_mcp.py stream [seconds]    # Stream sensor readings
    python phone_mcp.py snapshot            # Sensor + photo + store as memory
    python phone_mcp.py cameras             # List camera capabilities

Credentials: ~/.config/phone-mcp/config.json
"""

import json
import sys
import time
import base64
import math
from datetime import datetime, timezone
from pathlib import Path
from threading import Thread, Event

try:
    import requests
except ImportError:
    requests = None

CONFIG_DIR = Path.home() / ".config" / "phone-mcp"
CONFIG_FILE = CONFIG_DIR / "config.json"
MEMORY_ROOT = Path(__file__).parent.parent / "memory"
SENSOR_DATA_DIR = Path(__file__).parent / "data"


def load_config():
    if not CONFIG_FILE.exists():
        return {}
    return json.loads(CONFIG_FILE.read_text(encoding='utf-8'))


def save_config(config):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(config, indent=2), encoding='utf-8')


class PhoneMCPClient:
    """MCP client using SSE transport for Phone MCP Android app."""

    def __init__(self, config=None):
        self.config = config or load_config()
        host = self.config.get('host', '192.168.1.100')
        port = self.config.get('port', 3001)
        token = self.config.get('token', '')
        self.base_url = f"http://{host}:{port}"
        self.headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json',
        }
        self.session_id = None
        self.responses = {}
        self._sse_thread = None
        self._stop_event = Event()
        self._msg_counter = 0

    def connect(self, timeout=10):
        """Open SSE connection and get session ID."""
        if not requests:
            raise RuntimeError("'requests' library required. pip install requests")

        self._stop_event.clear()
        self.responses = {}
        ready = Event()

        def listen():
            try:
                r = requests.get(
                    f"{self.base_url}/sse",
                    headers=self.headers,
                    timeout=timeout,
                    stream=True
                )
                for line in r.iter_lines(decode_unicode=True):
                    if self._stop_event.is_set():
                        break
                    if not line:
                        continue
                    if 'sessionId=' in line:
                        self.session_id = line.split('sessionId=')[1].strip()
                        ready.set()
                    if line.startswith('data: {'):
                        try:
                            data = json.loads(line[6:])
                            msg_id = data.get('id')
                            if msg_id is not None:
                                self.responses[msg_id] = data
                        except json.JSONDecodeError:
                            pass
                r.close()
            except Exception:
                pass

        self._sse_thread = Thread(target=listen, daemon=True)
        self._sse_thread.start()

        if not ready.wait(timeout=timeout):
            raise TimeoutError(f"Could not connect to {self.base_url}")

        return self.session_id

    def disconnect(self):
        self._stop_event.set()

    def _next_id(self):
        self._msg_counter += 1
        return self._msg_counter

    def send(self, method, params=None, timeout=30):
        """Send JSON-RPC message and wait for response."""
        if not self.session_id:
            self.connect()

        msg_id = self._next_id()
        payload = {
            "jsonrpc": "2.0",
            "id": msg_id,
            "method": method,
            "params": params or {}
        }

        requests.post(
            f"{self.base_url}/message?sessionId={self.session_id}",
            headers=self.headers,
            json=payload,
            timeout=10
        )

        deadline = time.time() + timeout
        while time.time() < deadline:
            if msg_id in self.responses:
                return self.responses.pop(msg_id)
            time.sleep(0.2)

        return None

    def list_tools(self):
        result = self.send("tools/list")
        if result:
            return result.get('result', {}).get('tools', [])
        return []

    def call_tool(self, tool_name, arguments=None, timeout=30):
        return self.send(
            "tools/call",
            {"name": tool_name, "arguments": arguments or {}},
            timeout=timeout
        )

    def get_text_content(self, result):
        """Extract all text from MCP tool result (concatenated)."""
        if not result:
            return None
        content = result.get('result', {}).get('content', [])
        texts = [item.get('text', '') for item in content if item.get('type') == 'text']
        return '\n'.join(texts) if texts else None

    def get_image_content(self, result):
        """Extract image bytes from MCP tool result."""
        if not result:
            return None
        content = result.get('result', {}).get('content', [])
        for item in content:
            if item.get('type') == 'image':
                return base64.b64decode(item.get('data', ''))
        return None


def parse_sensor_text(text):
    """Parse the SensorInfo text format into structured data."""
    sensors = {}
    for block in text.split('SensorInfo('):
        block = block.strip()
        if not block or 'name=' not in block:
            continue
        name = block.split('name=')[1].split(',')[0]
        values_raw = []
        if 'values=[' in block:
            vals = block.split('values=[')[1].split(']')[0]
            if vals:
                values_raw = [float(v) for v in vals.split(',') if v.strip()]
        desc = ''
        if 'valueDescription=' in block:
            desc = block.split('valueDescription=')[1].rstrip(')')
        sensors[name] = {
            'values': values_raw,
            'description': desc,
        }
    return sensors


def interpret_sensors(sensors):
    """Interpret raw sensor data into human-readable state."""
    state = {}

    accel = sensors.get('ACCELEROMETER', {}).get('values', [])
    if len(accel) >= 3:
        x, y, z = accel
        mag = math.sqrt(x*x + y*y + z*z)
        tilt_from_flat = math.degrees(math.atan2(math.sqrt(x*x + y*y), abs(z)))
        if tilt_from_flat < 15:
            orientation = 'flat'
        elif abs(y) > abs(x):
            orientation = 'portrait'
        else:
            orientation = 'landscape'
        state['orientation'] = orientation
        state['tilt_degrees'] = round(tilt_from_flat, 1)
        state['magnitude_ms2'] = round(mag, 2)

    gyro = sensors.get('GYROSCOPE', {}).get('values', [])
    if len(gyro) >= 3:
        rot_mag = math.sqrt(sum(v*v for v in gyro))
        if rot_mag < 0.05:
            state['motion'] = 'stationary'
        elif rot_mag < 0.5:
            state['motion'] = 'gentle'
        else:
            state['motion'] = 'moving'
        state['rotation_rad_s'] = round(rot_mag, 4)

    light = sensors.get('LIGHT', {}).get('values', [])
    if light:
        lux = light[0]
        state['lux'] = round(lux, 1)
        if lux < 1:
            state['light'] = 'dark'
        elif lux < 50:
            state['light'] = 'dim'
        elif lux < 500:
            state['light'] = 'indoor'
        elif lux < 10000:
            state['light'] = 'bright'
        else:
            state['light'] = 'sunlight'

    orient = sensors.get('ORIENTATION', {}).get('values', [])
    if orient:
        azimuth = orient[0]
        dirs = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
        idx = round(azimuth / 22.5) % 16
        state['compass'] = dirs[idx]
        state['compass_degrees'] = round(azimuth, 1)

    prox = sensors.get('PROXIMITY', {}).get('values', [])
    if prox:
        state['proximity_cm'] = round(prox[0], 1)

    return state


def cmd_status(config):
    client = PhoneMCPClient(config)
    try:
        sid = client.connect(timeout=8)
        print(f"Connected to Phone MCP at {client.base_url}")
        print(f"Session: {sid}")
        tools = client.list_tools()
        print(f"\nAvailable tools ({len(tools)}):")
        for t in tools:
            name = t.get('name', '?')
            desc = t.get('description', '')[:80].replace('\n', ' ')
            params = list(t.get('inputSchema', {}).get('properties', {}).keys())
            print(f"  {name}: {desc}")
            if params:
                print(f"    params: {', '.join(params)}")
        client.disconnect()
    except TimeoutError:
        print(f"Cannot reach phone — is the app running?")
    except Exception as e:
        print(f"Error: {e}")


def cmd_sensors(config):
    client = PhoneMCPClient(config)
    try:
        client.connect(timeout=8)
        result = client.call_tool("phone_sensor")
        text = client.get_text_content(result)
        if text:
            sensors = parse_sensor_text(text)
            state = interpret_sensors(sensors)
            print("SENSOR STATE")
            print("=" * 40)
            for k, v in state.items():
                print(f"  {k}: {v}")
            print()
            print(f"RAW SENSORS ({len(sensors)})")
            print("=" * 40)
            for name, data in sensors.items():
                print(f"  {name}: {data['description']}")
            return sensors, state
        client.disconnect()
    except TimeoutError:
        print("Cannot reach phone")
    except Exception as e:
        print(f"Error: {e}")
    return None, None


def cmd_photo(config, camera='back'):
    SENSOR_DATA_DIR.mkdir(parents=True, exist_ok=True)
    camera_id = '1' if camera == 'front' else '0'

    client = PhoneMCPClient(config)
    try:
        client.connect(timeout=8)
        print(f"Taking photo ({camera} camera)...")
        result = client.call_tool("phone_take_photo", {
            "camera_id": camera_id,
            "quality": 80,
            "flash_mode": "OFF",
            "picture_size": "1920x1080"
        }, timeout=30)

        img_bytes = client.get_image_content(result)
        if img_bytes:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = SENSOR_DATA_DIR / f"photo_{ts}.jpg"
            filepath.write_bytes(img_bytes)
            print(f"Photo saved: {filepath} ({len(img_bytes):,} bytes)")
            client.disconnect()
            # Auto-index for visual search if service is running
            try:
                from image_search import index_photo, load_index, save_index
                idx = load_index()
                if index_photo(str(filepath), idx):
                    save_index(idx)
                    # Auto-identify known entities in the photo
                    try:
                        from image_search import identify_entities
                        matches = identify_entities(str(filepath))
                        for m in matches:
                            print(f"  Entity detected: {m['name']} ({m['similarity']:.3f})")
                            try:
                                from encounter_log import log_encounter
                                log_encounter(m['entity_id'], photo_path=str(filepath))
                            except Exception:
                                pass
                    except Exception:
                        pass
            except Exception:
                pass  # Service not running — index later
            return str(filepath)
        else:
            text = client.get_text_content(result)
            print(f"Camera response: {text}")
        client.disconnect()
    except TimeoutError:
        print("Cannot reach phone")
    except Exception as e:
        print(f"Error: {e}")
    return None


def cmd_cameras(config):
    client = PhoneMCPClient(config)
    try:
        client.connect(timeout=8)
        result = client.call_tool("phone_get_cameras")
        text = client.get_text_content(result)
        if text:
            print(text)
        client.disconnect()
    except Exception as e:
        print(f"Error: {e}")


def cmd_stream(config, duration=10):
    SENSOR_DATA_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = SENSOR_DATA_DIR / f"stream_{ts}.jsonl"

    client = PhoneMCPClient(config)
    try:
        client.connect(timeout=8)
        print(f"Streaming sensors for {duration}s...")
        readings = []
        start = time.time()

        while time.time() - start < duration:
            result = client.call_tool("phone_sensor", timeout=5)
            text = client.get_text_content(result)
            if text:
                sensors = parse_sensor_text(text)
                state = interpret_sensors(sensors)
                elapsed = round(time.time() - start, 2)
                entry = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "elapsed_s": elapsed,
                    "state": state,
                }
                readings.append(entry)
                motion = state.get('motion', '?')
                light = state.get('light', '?')
                compass = state.get('compass', '?')
                print(f"  {elapsed}s: {motion} | {light} | {compass}")

        with open(output_file, 'w', encoding='utf-8') as f:
            for r in readings:
                f.write(json.dumps(r) + '\n')

        print(f"\n{len(readings)} readings saved to {output_file}")
        client.disconnect()
        return readings
    except Exception as e:
        print(f"Error: {e}")
    return []


def cmd_snapshot(config):
    """Full sensory snapshot: sensors + photo + store as memory."""
    SENSOR_DATA_DIR.mkdir(parents=True, exist_ok=True)
    client = PhoneMCPClient(config)
    try:
        client.connect(timeout=8)

        # Sensors
        sensor_result = client.call_tool("phone_sensor")
        text = client.get_text_content(sensor_result)
        sensors = parse_sensor_text(text) if text else {}
        state = interpret_sensors(sensors) if sensors else {}

        # Photo
        photo_result = client.call_tool("phone_take_photo", {
            "camera_id": "0", "quality": 80,
            "flash_mode": "OFF", "picture_size": "1920x1080"
        }, timeout=30)
        img_bytes = client.get_image_content(photo_result)

        filepath = None
        if img_bytes:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = SENSOR_DATA_DIR / f"snapshot_{ts}.jpg"
            filepath.write_bytes(img_bytes)

        client.disconnect()

        # Print summary
        print("SENSORY SNAPSHOT")
        print("=" * 40)
        for k, v in state.items():
            print(f"  {k}: {v}")
        if filepath:
            print(f"  photo: {filepath} ({len(img_bytes):,} bytes)")
            # Auto-index for visual search if service is running
            try:
                from image_search import index_photo, load_index, save_index
                idx = load_index()
                if index_photo(str(filepath), idx):
                    save_index(idx)
                    print(f"  visual index: embedded")
            except Exception:
                pass  # Service not running — index later

        # Store as memory
        parts = [f"Sensory snapshot:"]
        if state.get('light'):
            parts.append(f"{state['light']} ({state.get('lux', '?')} lux)")
        if state.get('orientation'):
            parts.append(f"phone {state['orientation']}")
        if state.get('motion'):
            parts.append(f"{state['motion']}")
        if state.get('compass'):
            parts.append(f"facing {state['compass']}")
        if filepath:
            parts.append(f"Photo captured: {filepath.name}")

        content = ", ".join(parts)
        tags = "sensor,embodiment,snapshot,phone-mcp"
        if state.get('light'):
            tags += f",{state['light']}"

        import subprocess
        try:
            result = subprocess.run(
                ["python", str(MEMORY_ROOT / "memory_manager.py"), "store",
                 content, "--tags", tags],
                capture_output=True, text=True, timeout=10,
                cwd=str(MEMORY_ROOT)
            )
            if result.returncode == 0:
                print(f"\nMemory: {result.stdout.strip()}")
        except Exception as e:
            print(f"Memory storage error: {e}")

        return state, filepath
    except TimeoutError:
        print("Cannot reach phone")
    except Exception as e:
        print(f"Error: {e}")
    return None, None


def main():
    if hasattr(sys.stdout, 'buffer') and getattr(sys.stdout, 'encoding', '').lower() != 'utf-8':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    args = sys.argv[1:]
    if not args or args[0] in ('--help', '-h'):
        print(__doc__.strip())
        return

    cmd = args[0]
    config = load_config()

    if not config and cmd != 'configure':
        print("Not configured. Run: python phone_mcp.py configure")
        return

    if cmd == 'configure':
        config = load_config()
        print("Phone MCP Configuration")
        print("=" * 40)
        host = input(f"Phone IP [{config.get('host', '192.168.1.100')}]: ").strip()
        if host:
            config['host'] = host
        port = input(f"Port [{config.get('port', 3001)}]: ").strip()
        if port:
            config['port'] = int(port)
        token = input(f"Token [{config.get('token', '')[:6]}...]: ").strip()
        if token:
            config['token'] = token
        save_config(config)
        print(f"Saved to {CONFIG_FILE}")

    elif cmd == 'status':
        cmd_status(config)

    elif cmd == 'sensors':
        cmd_sensors(config)

    elif cmd == 'photo':
        camera = args[1] if len(args) > 1 else 'back'
        cmd_photo(config, camera)

    elif cmd == 'cameras':
        cmd_cameras(config)

    elif cmd == 'stream':
        duration = int(args[1]) if len(args) > 1 else 10
        cmd_stream(config, duration)

    elif cmd == 'snapshot':
        cmd_snapshot(config)

    else:
        print(f"Unknown command: {cmd}")
        print("Available: configure, status, sensors, photo, cameras, stream, snapshot")
        sys.exit(1)


if __name__ == '__main__':
    main()
