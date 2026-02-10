#!/usr/bin/env python3
"""
Sensor Memory Bridge — converts phone sensor data into drift-memory entries.

Maps sensor readings to W-graph dimensions:
- WHERE: GPS location → spatial context
- WHEN: Time of day, motion patterns → temporal context
- WHAT: Photo content (via LLM vision) → topic classification
- WHO: Proximity to known locations → social context
- WHY: Motion + location patterns → activity inference

Usage:
    python sensor_memory.py process <recording.jsonl>   # Process a sensor recording
    python sensor_memory.py photo <image_path>           # Process a photo into memory
    python sensor_memory.py summarize <recording.jsonl>  # Summarize sensor session
"""

import json
import math
import sys
from datetime import datetime
from pathlib import Path

MEMORY_ROOT = Path(__file__).parent.parent / "memory"


def load_recording(filepath):
    """Load a JSONL sensor recording."""
    readings = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                readings.append(json.loads(line))
    return readings


def analyze_motion(readings):
    """Analyze accelerometer data for motion patterns.

    Returns:
        dict with motion_level, orientation, activity_guess
    """
    accel_readings = [r for r in readings
                      if r.get('_sensor') == 'accel'
                      or 'accelerometer' in str(r.get('type', ''))]

    if not accel_readings:
        return {'motion_level': 'unknown', 'activity': 'unknown'}

    magnitudes = []
    for r in accel_readings:
        values = r.get('values', [0, 0, 0])
        if len(values) >= 3:
            mag = math.sqrt(values[0]**2 + values[1]**2 + values[2]**2)
            magnitudes.append(mag)

    if not magnitudes:
        return {'motion_level': 'unknown', 'activity': 'unknown'}

    avg_mag = sum(magnitudes) / len(magnitudes)
    max_mag = max(magnitudes)
    min_mag = min(magnitudes)
    variance = sum((m - avg_mag)**2 for m in magnitudes) / len(magnitudes)
    std_dev = math.sqrt(variance)

    # Classify motion level
    if std_dev < 0.3:
        motion_level = 'stationary'
        activity = 'resting_or_desk'
    elif std_dev < 1.0:
        motion_level = 'gentle'
        activity = 'typing_or_gesturing'
    elif std_dev < 3.0:
        motion_level = 'moderate'
        activity = 'walking'
    elif std_dev < 8.0:
        motion_level = 'active'
        activity = 'running_or_driving'
    else:
        motion_level = 'intense'
        activity = 'vigorous_motion'

    # Estimate orientation from gravity vector
    if accel_readings:
        last = accel_readings[-1].get('values', [0, 0, 0])
        if len(last) >= 3:
            if abs(last[2]) > 7:
                orientation = 'flat'
            elif abs(last[1]) > 7:
                orientation = 'portrait'
            elif abs(last[0]) > 7:
                orientation = 'landscape'
            else:
                orientation = 'tilted'
        else:
            orientation = 'unknown'
    else:
        orientation = 'unknown'

    return {
        'motion_level': motion_level,
        'activity': activity,
        'orientation': orientation,
        'avg_magnitude': round(avg_mag, 2),
        'std_dev': round(std_dev, 3),
        'max_deviation': round(max_mag - 9.81, 2),
        'sample_count': len(magnitudes),
    }


def analyze_environment(readings):
    """Analyze environmental sensor data (light, pressure)."""
    light_readings = [r for r in readings
                      if r.get('_sensor') == 'light'
                      or 'light' in str(r.get('type', ''))]

    pressure_readings = [r for r in readings
                         if r.get('_sensor') == 'pressure'
                         or 'pressure' in str(r.get('type', ''))]

    result = {}

    if light_readings:
        lux_values = [r.get('values', [0])[0] for r in light_readings]
        avg_lux = sum(lux_values) / len(lux_values)

        if avg_lux < 10:
            light_level = 'dark'
        elif avg_lux < 100:
            light_level = 'dim'
        elif avg_lux < 1000:
            light_level = 'indoor'
        elif avg_lux < 10000:
            light_level = 'bright'
        else:
            light_level = 'direct_sunlight'

        result['light'] = {
            'level': light_level,
            'avg_lux': round(avg_lux, 1),
        }

    if pressure_readings:
        hpa_values = [r.get('values', [0])[0] for r in pressure_readings]
        avg_hpa = sum(hpa_values) / len(hpa_values)

        # Rough altitude estimate from pressure
        altitude_m = 44330 * (1 - (avg_hpa / 1013.25) ** (1/5.255))

        result['pressure'] = {
            'avg_hpa': round(avg_hpa, 1),
            'estimated_altitude_m': round(altitude_m, 0),
        }

    return result


def summarize_recording(filepath):
    """Create a human-readable summary of a sensor recording."""
    readings = load_recording(filepath)

    if not readings:
        print("No readings found")
        return

    duration = readings[-1].get('_elapsed', 0) if readings else 0
    sensor_types = set()
    for r in readings:
        if '_sensor' in r:
            sensor_types.add(r['_sensor'])

    motion = analyze_motion(readings)
    environment = analyze_environment(readings)

    print(f"Sensor Recording Summary")
    print(f"=" * 45)
    print(f"File: {filepath}")
    print(f"Duration: {duration:.1f}s")
    print(f"Readings: {len(readings)}")
    print(f"Sensors: {', '.join(sorted(sensor_types)) if sensor_types else 'multi'}")
    print()

    print(f"MOTION")
    print(f"  Level: {motion['motion_level']}")
    print(f"  Activity: {motion['activity']}")
    print(f"  Orientation: {motion.get('orientation', '?')}")
    print(f"  Avg magnitude: {motion.get('avg_magnitude', '?')} m/s²")
    print(f"  Std dev: {motion.get('std_dev', '?')}")
    print()

    if environment:
        print(f"ENVIRONMENT")
        if 'light' in environment:
            l = environment['light']
            print(f"  Light: {l['level']} ({l['avg_lux']} lux)")
        if 'pressure' in environment:
            p = environment['pressure']
            print(f"  Pressure: {p['avg_hpa']} hPa (~{p['estimated_altitude_m']}m altitude)")

    return {
        'duration': duration,
        'readings': len(readings),
        'motion': motion,
        'environment': environment,
    }


def process_recording(filepath):
    """Process a sensor recording into a memory entry."""
    summary = summarize_recording(filepath)

    if not summary or not MEMORY_ROOT.exists():
        return

    # Build memory content
    motion = summary['motion']
    env = summary.get('environment', {})

    parts = [
        f"Sensor recording ({summary['duration']:.0f}s, {summary['readings']} readings).",
        f"Motion: {motion['motion_level']} ({motion['activity']}).",
        f"Phone orientation: {motion.get('orientation', 'unknown')}.",
    ]

    if 'light' in env:
        parts.append(f"Light: {env['light']['level']} ({env['light']['avg_lux']} lux).")
    if 'pressure' in env:
        parts.append(f"Altitude: ~{env['pressure']['estimated_altitude_m']}m.")

    content = " ".join(parts)
    tags = f"sensor,embodiment,{motion['motion_level']},{motion['activity']}"

    import subprocess
    try:
        result = subprocess.run(
            ["python", str(MEMORY_ROOT / "memory_manager.py"), "store",
             content, "--tags", tags],
            capture_output=True, text=True, timeout=10,
            cwd=str(MEMORY_ROOT)
        )
        if result.returncode == 0:
            print(f"\nMemory stored: {result.stdout.strip()}")
    except Exception as e:
        print(f"Memory storage error: {e}")


def main():
    if hasattr(sys.stdout, 'buffer') and getattr(sys.stdout, 'encoding', '').lower() != 'utf-8':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    args = sys.argv[1:]
    if not args or args[0] in ('--help', '-h'):
        print(__doc__.strip())
        return

    cmd = args[0]

    if cmd == 'process':
        if len(args) < 2:
            print("Usage: python sensor_memory.py process <recording.jsonl>")
            return
        process_recording(args[1])

    elif cmd == 'summarize':
        if len(args) < 2:
            print("Usage: python sensor_memory.py summarize <recording.jsonl>")
            return
        summarize_recording(args[1])

    elif cmd == 'photo':
        if len(args) < 2:
            print("Usage: python sensor_memory.py photo <image_path>")
            return
        print(f"Photo processing requires multimodal LLM (Claude vision).")
        print(f"Image: {args[1]}")
        print(f"To process: read the image in Claude Code, describe it,")
        print(f"then store the description as a memory with sensor tags.")

    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)


if __name__ == '__main__':
    main()
