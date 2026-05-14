# shellcheck shell=bash
# Claude Code — native install (no node/sudo).

step "04 — Claude Code (native)"

if [ -x "$LOCAL_BIN/claude" ]; then
    ok "Claude Code already at $LOCAL_BIN/claude ($(claude --version 2>/dev/null))"
elif have claude; then
    log "Claude found at $(which claude) — switching to native install"
    claude install || warn "claude install failed — try manually"
else
    log "No claude binary present. Installing via official native installer."
    # Official one-shot installer (no node required)
    curl -fsSL https://claude.ai/install.sh | bash || warn "fallback: install via npm or download release manually"
fi

# Re-resolve PATH lookup
hash -r 2>/dev/null || true

if [ -x "$LOCAL_BIN/claude" ]; then
    ok "Claude Code: $($LOCAL_BIN/claude --version 2>/dev/null)"
else
    warn "Claude Code binary not at $LOCAL_BIN/claude — verify install path"
fi
