#!/usr/bin/env python3
"""
SensorServer WebSocket Client — continuous sensor streaming from Android.

Connects to UmerCodez/SensorServer app for real-time sensor data.
Processes readings into memory-compatible events.

Usage:
    python sensor_stream.py connect                    # Test connection
    python sensor_stream.py stream [seconds]           # Stream all sensors
    python sensor_stream.py record [seconds]           # Stream + save to file
    python sensor_stream.py motion [seconds]           # Motion detection mode
    python sensor_stream.py configure                  # Set phone IP/port

WebSocket endpoints:
    ws://<ip>:<port>/sensor/connect?type=android.sensor.accelerometer
    ws://<ip>:<port>/sensor/connect?type=android.sensor.gyroscope
    ws://<ip>:<port>/sensor/connect?type=android.sensor.light
    ws://<ip>:<port>/sensor/connect?type=android.sensor.pressure
    ws://<ip>:<port>/gps
    ws://<ip>:<port>/touchscreen
    ws://<ip>:<port>/sensors/connect?types=["type1","type2"]

Credentials: ~/.config/sensor-server/config.json
"""

import json
import sys
import time
import math
from datetime import datetime, timezone
from pathlib import Path
from threading import Thread, Event

# Optional imports
try:
    import websocket
except ImportError:
    websocket = None


CONFIG_DIR = Path.home() / ".config" / "sensor-server"
CONFIG_FILE = CONFIG_DIR / "config.json"
SENSOR_DATA_DIR = Path(__file__).parent / "data"

# Common sensor types
SENSORS = {
    'accel': 'android.sensor.accelerometer',
    'gyro': 'android.sensor.gyroscope',
    'mag': 'android.sensor.magnetic_field',
    'light': 'android.sensor.light',
    'pressure': 'android.sensor.pressure',
    'proximity': 'android.sensor.proximity',
    'gravity': 'android.sensor.gravity',
    'rotation': 'android.sensor.rotation_vector',
    'step': 'android.sensor.step_counter',
}


def load_config():
    if not CONFIG_FILE.exists():
        return {}
    return json.loads(CONFIG_FILE.read_text(encoding='utf-8'))


def save_config(config):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(config, indent=2), encoding='utf-8')


def get_ws_url(config, sensor_type):
    host = config.get('host', '192.168.1.100')
    port = config.get('port', 8080)
    return f"ws://{host}:{port}/sensor/connect?type={sensor_type}"


def get_multi_ws_url(config, sensor_types):
    host = config.get('host', '192.168.1.100')
    port = config.get('port', 8080)
    types_json = json.dumps(sensor_types)
    return f"ws://{host}:{port}/sensors/connect?types={types_json}"


def get_gps_url(config):
    host = config.get('host', '192.168.1.100')
    port = config.get('port', 8080)
    return f"ws://{host}:{port}/gps"


def test_connection(config):
    """Test WebSocket connection to SensorServer."""
    if not websocket:
        print("ERROR: 'websocket-client' library required. pip install websocket-client")
        return False

    url = get_ws_url(config, SENSORS['accel'])
    print(f"Connecting to {url}...")

    try:
        ws = websocket.create_connection(url, timeout=5)
        data = ws.recv()
        reading = json.loads(data)
        ws.close()
        print(f"Connected! Sample reading:")
        print(f"  values: {reading.get('values', [])}")
        print(f"  accuracy: {reading.get('accuracy', '?')}")
        print(f"  timestamp: {reading.get('timestamp', '?')}")
        return True
    except Exception as e:
        print(f"Connection failed: {e}")
        print(f"Is SensorServer running on the phone?")
        return False


def stream_sensor(config, sensor_key='accel', duration=10, callback=None):
    """Stream a single sensor for N seconds."""
    if not websocket:
        print("ERROR: 'websocket-client' required")
        return []

    sensor_type = SENSORS.get(sensor_key, sensor_key)
    url = get_ws_url(config, sensor_type)

    readings = []
    start = time.time()

    try:
        ws = websocket.create_connection(url, timeout=5)
        print(f"Streaming {sensor_key} for {duration}s...")

        while time.time() - start < duration:
            data = ws.recv()
            reading = json.loads(data)
            reading['_sensor'] = sensor_key
            reading['_received'] = datetime.now(timezone.utc).isoformat()
            reading['_elapsed'] = round(time.time() - start, 3)
            readings.append(reading)

            if callback:
                callback(reading)

        ws.close()
    except Exception as e:
        print(f"Stream error: {e}")

    return readings


def stream_multi(config, sensor_keys=None, duration=10, callback=None):
    """Stream multiple sensors simultaneously."""
    if not websocket:
        print("ERROR: 'websocket-client' required")
        return []

    if sensor_keys is None:
        sensor_keys = ['accel', 'gyro', 'light']

    sensor_types = [SENSORS.get(k, k) for k in sensor_keys]
    url = get_multi_ws_url(config, sensor_types)

    readings = []
    start = time.time()

    try:
        ws = websocket.create_connection(url, timeout=5)
        print(f"Streaming {', '.join(sensor_keys)} for {duration}s...")

        while time.time() - start < duration:
            data = ws.recv()
            reading = json.loads(data)
            reading['_received'] = datetime.now(timezone.utc).isoformat()
            reading['_elapsed'] = round(time.time() - start, 3)
            readings.append(reading)

            if callback:
                callback(reading)

        ws.close()
    except Exception as e:
        print(f"Stream error: {e}")

    return readings


def detect_motion(config, duration=30, threshold=2.0):
    """Motion detection mode — alert on significant movement.

    Monitors accelerometer and flags readings where total acceleration
    deviates significantly from gravity (~9.8 m/s²).
    """
    if not websocket:
        print("ERROR: 'websocket-client' required")
        return

    url = get_ws_url(config, SENSORS['accel'])
    events = []
    start = time.time()

    try:
        ws = websocket.create_connection(url, timeout=5)
        print(f"Motion detection for {duration}s (threshold: {threshold} m/s²)...")
        print(f"Monitoring accelerometer for deviations from gravity...")
        print()

        baseline_g = 9.81
        count = 0
        motion_count = 0

        while time.time() - start < duration:
            data = ws.recv()
            reading = json.loads(data)
            values = reading.get('values', [0, 0, 0])

            if len(values) >= 3:
                x, y, z = values[0], values[1], values[2]
                magnitude = math.sqrt(x*x + y*y + z*z)
                deviation = abs(magnitude - baseline_g)

                count += 1

                if deviation > threshold:
                    motion_count += 1
                    elapsed = round(time.time() - start, 1)
                    event = {
                        'timestamp': datetime.now(timezone.utc).isoformat(),
                        'elapsed': elapsed,
                        'magnitude': round(magnitude, 2),
                        'deviation': round(deviation, 2),
                        'x': round(x, 2),
                        'y': round(y, 2),
                        'z': round(z, 2),
                    }
                    events.append(event)
                    print(f"  MOTION @ {elapsed}s: mag={magnitude:.2f} dev={deviation:.2f} (x={x:.1f} y={y:.1f} z={z:.1f})")

        ws.close()

        print(f"\nSummary: {motion_count} motion events in {count} readings ({duration}s)")
        if motion_count > 0:
            print(f"Motion rate: {motion_count/count*100:.1f}%")

    except Exception as e:
        print(f"Motion detection error: {e}")

    return events


def record_to_file(config, duration=10, sensor_keys=None):
    """Record sensor data to JSONL file."""
    SENSOR_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if sensor_keys is None:
        sensor_keys = ['accel', 'gyro', 'light']

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = SENSOR_DATA_DIR / f"recording_{timestamp}.jsonl"

    readings = stream_multi(config, sensor_keys, duration)

    with open(output_file, 'w', encoding='utf-8') as f:
        for r in readings:
            f.write(json.dumps(r) + '\n')

    print(f"\nRecorded {len(readings)} readings to {output_file}")
    return output_file


def configure():
    """Interactive setup."""
    config = load_config()

    print("SensorServer Configuration")
    print("=" * 40)
    print()
    print("1. Install 'Sensor Server' from F-Droid or GitHub")
    print("   https://github.com/UmerCodez/SensorServer")
    print("2. Open the app, start the server")
    print("3. Note the IP address shown in the app")
    print()

    host = input(f"Phone IP address [{config.get('host', '192.168.1.100')}]: ").strip()
    if host:
        config['host'] = host

    port = input(f"Port [{config.get('port', 8080)}]: ").strip()
    if port:
        config['port'] = int(port)

    save_config(config)
    print(f"\nConfig saved to {CONFIG_FILE}")
    print("Testing connection...")
    test_connection(config)


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

    if cmd == 'configure':
        configure()

    elif cmd == 'connect':
        if not config:
            print("Not configured. Run: python sensor_stream.py configure")
            return
        test_connection(config)

    elif cmd == 'stream':
        if not config:
            print("Not configured.")
            return
        duration = int(args[1]) if len(args) > 1 else 10
        readings = stream_multi(config, duration=duration,
                               callback=lambda r: print(f"  {r.get('_elapsed', '?')}s: {r.get('values', [])[:3]}"))
        print(f"\nTotal: {len(readings)} readings")

    elif cmd == 'record':
        if not config:
            print("Not configured.")
            return
        duration = int(args[1]) if len(args) > 1 else 10
        record_to_file(config, duration)

    elif cmd == 'motion':
        if not config:
            print("Not configured.")
            return
        duration = int(args[1]) if len(args) > 1 else 30
        threshold = float(args[2]) if len(args) > 2 else 2.0
        detect_motion(config, duration, threshold)

    else:
        print(f"Unknown command: {cmd}")
        print("Available: configure, connect, stream, record, motion")
        sys.exit(1)


if __name__ == '__main__':
    main()
