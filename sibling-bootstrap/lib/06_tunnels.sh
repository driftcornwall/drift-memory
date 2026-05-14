# shellcheck shell=bash
# Socat tunnels: forward host services as localhost on the VM, persist via crontab.

step "07 — Socat tunnels (host services → localhost) + @reboot crontab"

TUNNELS="$LOCAL_BIN/${AGENT_NAME}_tunnels"
log "Writing tunnel script: $TUNNELS"
template_render "$TEMPLATES_DIR/agent_tunnels.template" "$TUNNELS"
chmod +x "$TUNNELS"

# Run it now
log "Starting tunnels (NLI:$NLI_PORT, consd:$CONSD_PORT, ollama:$OLLAMA_PORT)"
"$TUNNELS"
sleep 1

for p in "$NLI_PORT" "$CONSD_PORT" "$OLLAMA_PORT"; do
    if ss -tlnp 2>/dev/null | grep -q ":$p "; then
        ok "tunnel up on :$p"
    else
        warn "tunnel for :$p not detected — check /tmp/socat-*.log"
    fi
done

# Add to crontab if not already there
CRON_LINE="@reboot $TUNNELS"
if crontab -l 2>/dev/null | grep -qF "$CRON_LINE"; then
    ok "Crontab @reboot already present"
else
    log "Adding @reboot entry to crontab"
    ( crontab -l 2>/dev/null; echo "$CRON_LINE" ) | crontab -
    ok "Crontab updated"
fi
