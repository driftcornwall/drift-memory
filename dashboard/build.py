"""Bundle data.json inline into index.html for standalone file:// usage."""
import sys
import json
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

HERE = Path(__file__).parent

with open(HERE / 'data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

with open(HERE / 'index.html', 'r', encoding='utf-8') as f:
    html = f.read()

compact = json.dumps(data, separators=(',', ':'), ensure_ascii=False)

# Insert inline data script before the main script
inject = f'<script>const INLINE_DATA = {compact};</script>\n'
html = html.replace('<script>\nconst DIM_COLORS', inject + '<script>\nconst DIM_COLORS')

out = HERE / 'drift-dashboard.html'
with open(out, 'w', encoding='utf-8') as f:
    f.write(html)

size_kb = out.stat().st_size / 1024
print(f'Built: {out} ({size_kb:.0f} KB)')
