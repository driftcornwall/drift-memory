#!/usr/bin/env python3
"""Claude Code statusLine — Parrot PS1 style with token + context display."""
import json, os, sys, socket, getpass

raw = sys.stdin.read()
try:
    with open("/tmp/statusline-last-input.json", "w") as f:
        f.write(raw)
except Exception:
    pass

try:
    d = json.loads(raw) if raw.strip() else {}
except Exception:
    d = {}

def g(obj, *path, default=None):
    for k in path:
        if isinstance(obj, dict) and k in obj:
            obj = obj[k]
        else:
            return default
    return obj

cwd   = g(d, "cwd") or g(d, "workspace", "current_dir") or os.getcwd()
model = g(d, "model", "display_name") or g(d, "model", "id") or ""

ctx_pct   = g(d, "context_window", "used_percentage")
ctx_used  = g(d, "context_window", "used_tokens") or g(d, "context_window", "used")
ctx_total = g(d, "context_window", "capacity")    or g(d, "context_window", "total")

in_tok      = g(d, "cost", "input_tokens")          or g(d, "tokens", "input")
out_tok     = g(d, "cost", "output_tokens")         or g(d, "tokens", "output")
cache_read  = g(d, "cost", "cache_read_tokens")     or g(d, "tokens", "cache_read")
cost_usd    = g(d, "cost", "total_cost_usd")

home = os.path.expanduser("~")
if cwd.startswith(home):
    cwd = "~" + cwd[len(home):]

RED, WHITE, YELLOW, CYAN, GREEN, DIM_YEL, RESET = (
    "\033[0;31m","\033[0;39m","\033[01;33m","\033[01;96m",
    "\033[0;32m","\033[0;33m","\033[0m",
)

def fmt_tok(n):
    if n is None: return None
    n = int(n)
    if n >= 1_000_000: return f"{n/1_000_000:.1f}M"
    if n >= 1_000:     return f"{n/1_000:.1f}k"
    return str(n)

user = getpass.getuser()
host = socket.gethostname().split(".")[0]
left = f"{RED}┌─[{WHITE}{user}{YELLOW}@{CYAN}{host}{RED}]─[{GREEN}{cwd}{RED}]{RESET}"

parts = [model] if model else []
if ctx_pct is not None:
    parts.append(f"ctx:{int(round(float(ctx_pct)))}%")
elif ctx_used and ctx_total:
    parts.append(f"ctx:{fmt_tok(ctx_used)}/{fmt_tok(ctx_total)}")

tok_bits = []
if in_tok     is not None: tok_bits.append(f"in:{fmt_tok(in_tok)}")
if out_tok    is not None: tok_bits.append(f"out:{fmt_tok(out_tok)}")
if cache_read is not None: tok_bits.append(f"cache:{fmt_tok(cache_read)}")
if tok_bits:
    parts.append(" ".join(tok_bits))
if cost_usd is not None:
    parts.append(f"${float(cost_usd):.3f}")

right = f"{DIM_YEL}{'  '.join(parts)}{RESET}" if parts else ""
sys.stdout.write(f"{left}  {right}")
