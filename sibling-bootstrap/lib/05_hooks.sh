# shellcheck shell=bash
# Claude Code hooks: agent_hook wrapper + settings.json + statusline.

step "06 — Claude Code hooks + settings.json + statusline"

ensure_dir "$CLAUDE_HOOKS_DIR"
ensure_dir "$CLAUDE_DIR"

# ---- agent_hook wrapper ----
HOOK_WRAPPER="$CLAUDE_HOOKS_DIR/${AGENT_NAME}_hook"
log "Writing hook wrapper: $HOOK_WRAPPER"
template_render "$TEMPLATES_DIR/agent_hook.template" "$HOOK_WRAPPER"
chmod +x "$HOOK_WRAPPER"
ok "Wrapper sources .env, execs uv on hooks/<script>"

# ---- statusline ----
STATUSLINE="$CLAUDE_DIR/statusline-command.sh"
log "Installing statusline: $STATUSLINE"
cp "$TEMPLATES_DIR/statusline.py" "$STATUSLINE"
chmod +x "$STATUSLINE"
ok "Statusline written (python-native, no jq)"

# ---- settings.json ----
SETTINGS="$CLAUDE_DIR/settings.json"
if [ -f "$SETTINGS" ]; then
    BACKUP="$SETTINGS.pre-bootstrap.$(date +%s).bak"
    cp "$SETTINGS" "$BACKUP"
    log "Existing settings.json backed up → $BACKUP"
fi

log "Writing settings.json with all 7 hook events"
template_render "$TEMPLATES_DIR/settings.json.template" "$SETTINGS"

# Validate JSON
python3 -c "import json; json.load(open('$SETTINGS'))" \
    && ok "settings.json valid JSON" \
    || die "settings.json is not valid JSON — restore from $BACKUP"
