# shellcheck shell=bash
# Clone drift-memory, write .env, create Postgres schema, install agent wrapper.

step "05 — drift-memory repo + .env + DB schema"

# ---- clone repo ----
if [ -d "$MEMORY_DIR/.git" ]; then
    ok "Memory repo already at $MEMORY_DIR (skipping clone)"
    log "Pulling latest"
    git -C "$MEMORY_DIR" pull --ff-only 2>&1 | tail -3 || warn "git pull had issues"
else
    log "Cloning $MEMORY_REPO → $MEMORY_DIR"
    ensure_dir "$(dirname "$MEMORY_DIR")"
    git clone --depth 50 "$MEMORY_REPO" "$MEMORY_DIR"
fi

# ---- write .env ----
log "Writing $MEMORY_DIR/.env"
template_render "$TEMPLATES_DIR/env.template" "$MEMORY_DIR/.env"
chmod 600 "$MEMORY_DIR/.env"
ok ".env written (AGENT_SCHEMA=$AGENT_NAME)"

# ---- agent wrapper script ($AGENT_NAME → ~/.local/bin/<agent>) ----
log "Installing agent wrapper at $LOCAL_BIN/$AGENT_NAME"
template_render "$TEMPLATES_DIR/agent_wrapper.template" "$LOCAL_BIN/$AGENT_NAME"
chmod +x "$LOCAL_BIN/$AGENT_NAME"
ok "Wrapper: $LOCAL_BIN/$AGENT_NAME (sources .env, runs scripts via uv)"

# ---- DB connectivity ----
log "Testing Postgres connection ($HOST_IP:$DB_PORT)"
if ! timeout 5 bash -c "</dev/tcp/$HOST_IP/$DB_PORT" 2>/dev/null; then
    die "Postgres unreachable at $HOST_IP:$DB_PORT — check host firewall + service"
fi
ok "Postgres TCP reachable"

# ---- ensure create_agent_schema function exists ----
HAS_FN=$(pg -tAc "SELECT 1 FROM pg_proc WHERE proname='create_agent_schema' LIMIT 1;" 2>/dev/null || echo "")
if [ "$HAS_FN" != "1" ]; then
    log "Installing schema.sql (provides create_agent_schema function)"
    pg -f "$MEMORY_DIR/docs/schema.sql" || die "Failed to install schema.sql"
fi
ok "create_agent_schema() available"

# ---- create the new agent's schema ----
EXISTING=$(pg -tAc "SELECT 1 FROM information_schema.schemata WHERE schema_name='$AGENT_NAME';" 2>/dev/null || echo "")
if [ "$EXISTING" = "1" ]; then
    ok "Schema '$AGENT_NAME' already exists in $DB_NAME"
else
    log "Creating schema '$AGENT_NAME'"
    pg -c "SELECT create_agent_schema('$AGENT_NAME');"
    ok "Schema '$AGENT_NAME' created"
fi

TBL_COUNT=$(pg -tAc "SELECT count(*) FROM information_schema.tables WHERE table_schema='$AGENT_NAME';")
ok "Tables in $AGENT_NAME schema: $TBL_COUNT"
