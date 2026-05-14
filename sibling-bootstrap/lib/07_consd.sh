# shellcheck shell=bash
# Consolidation daemon — local docker container for this VM only.

step "08 — Consolidation daemon (docker container, port $CONSD_PORT)"

CONSD_DIR="$MEMORY_DIR/consolidation-daemon"
if ! [ -d "$CONSD_DIR" ]; then
    warn "$CONSD_DIR not found — repo layout may have changed; skipping consd"
    return 0
fi

if ! have docker; then
    warn "docker not available — skipping consd. Install docker.io and re-run."
    return 0
fi

# Make sure user can talk to docker daemon
if ! docker ps >/dev/null 2>&1; then
    warn "docker daemon not reachable as $USER (group not yet active?). Skipping consd."
    warn "After 'newgrp docker' or relogin, run: cd $CONSD_DIR && docker compose up -d"
    return 0
fi

cd "$CONSD_DIR" || die "cannot cd $CONSD_DIR"

# Compose may need .env with AGENT_NAME if it's parameterized
if [ -f docker-compose.yml ] && grep -q "AGENT_SCHEMA\|AGENT_NAME" docker-compose.yml; then
    cat > .env <<EOF
AGENT_NAME=$AGENT_NAME
AGENT_SCHEMA=$AGENT_NAME
DB_HOST=$HOST_IP
DB_PORT=$DB_PORT
DB_NAME=$DB_NAME
DB_USER=$DB_USER
DB_PASSWORD=$DB_PASSWORD
EOF
    chmod 600 .env
fi

log "docker compose up -d"
docker compose up -d 2>&1 | tail -10 || warn "compose up had errors"

sleep 3
if curl -sf -m 3 "http://localhost:$CONSD_PORT/health" >/dev/null 2>&1; then
    ok "Consolidation daemon healthy on :$CONSD_PORT"
else
    warn "Consd not responding yet on :$CONSD_PORT — check 'docker compose logs'"
fi
