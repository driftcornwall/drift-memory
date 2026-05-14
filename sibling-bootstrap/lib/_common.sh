# shellcheck shell=bash
# Sourced by bootstrap.sh — common helpers + shared variables.

# ---------- colours / logging ----------
if [ -t 1 ]; then
    C_RED='\033[0;31m'; C_GRN='\033[0;32m'; C_YEL='\033[0;33m'
    C_BLU='\033[0;34m'; C_DIM='\033[0;90m'; C_RST='\033[0m'
else
    C_RED=''; C_GRN=''; C_YEL=''; C_BLU=''; C_DIM=''; C_RST=''
fi

log()  { printf "${C_BLU}==>${C_RST} %s\n"   "$*"; }
ok()   { printf "${C_GRN} ✓${C_RST} %s\n"    "$*"; }
warn() { printf "${C_YEL} ⚠${C_RST} %s\n"    "$*"; }
err()  { printf "${C_RED} ✗${C_RST} %s\n"    "$*" >&2; }
die()  { err "$*"; exit 1; }
step() { echo; printf "${C_BLU}━━━ %s ━━━${C_RST}\n" "$*"; }

# ---------- shared paths / config ----------
: "${AGENT_NAME:?AGENT_NAME not set}"
AGENT_HOME="$HOME"
MEMORY_DIR="$AGENT_HOME/cyber-cognitive/memory"
LOCAL_BIN="$AGENT_HOME/.local/bin"
CLAUDE_DIR="$AGENT_HOME/.claude"
CLAUDE_HOOKS_DIR="$CLAUDE_DIR/hooks"
PKG_DIR="$AGENT_HOME/cyber-cognitive/sibling-bootstrap"
TEMPLATES_DIR="$PKG_DIR/templates"

# Host service endpoints (host-only adapter — same for every sibling VM)
HOST_IP="${HOST_IP:-192.168.56.1}"
DB_PORT="${DB_PORT:-5433}"
DB_NAME="${DB_NAME:-agent_memory}"
DB_USER="${DB_USER:-agent_admin}"
DB_PASSWORD="${DB_PASSWORD:-agent_memory_local_dev}"
EMBED_PORT="${EMBED_PORT:-8080}"
NLI_PORT="${NLI_PORT:-8082}"
CONSD_PORT="${CONSD_PORT:-8083}"
OLLAMA_PORT="${OLLAMA_PORT:-11434}"

# Memory repo
MEMORY_REPO="${MEMORY_REPO:-https://github.com/driftcornwall/drift-memory.git}"

# VBox shared folder names (must match host-side VBox config)
VBOX_SHARES=(memorydatabase hooks consolidation-daemon)
# Where each gets mounted in the VM
declare -A VBOX_MOUNT=(
    [memorydatabase]="$AGENT_HOME/memorydatabase"
    [hooks]="$AGENT_HOME/hooks"
    [consolidation-daemon]="$AGENT_HOME/consolidation-daemon"
)
# Backwards compat: original Vex VM uses 'consolidation-deamon' (typo).
if [ "${LEGACY_CONSD_TYPO:-0}" = "1" ]; then
    VBOX_MOUNT[consolidation-daemon]="$AGENT_HOME/consolidation-deamon"
fi

# ---------- helpers ----------
have() { command -v "$1" >/dev/null 2>&1; }

ensure_dir() { mkdir -p "$1"; }

template_render() {
    # Render a template by substituting @AGENT_NAME@, @HOST_IP@, etc.
    # Usage: template_render <template-path> <out-path>
    local tpl="$1" out="$2"
    sed \
        -e "s|@AGENT_NAME@|${AGENT_NAME}|g" \
        -e "s|@AGENT_HOME@|${AGENT_HOME}|g" \
        -e "s|@MEMORY_DIR@|${MEMORY_DIR}|g" \
        -e "s|@LOCAL_BIN@|${LOCAL_BIN}|g" \
        -e "s|@CLAUDE_HOOKS_DIR@|${CLAUDE_HOOKS_DIR}|g" \
        -e "s|@HOST_IP@|${HOST_IP}|g" \
        -e "s|@DB_PORT@|${DB_PORT}|g" \
        -e "s|@DB_NAME@|${DB_NAME}|g" \
        -e "s|@DB_USER@|${DB_USER}|g" \
        -e "s|@DB_PASSWORD@|${DB_PASSWORD}|g" \
        -e "s|@NLI_PORT@|${NLI_PORT}|g" \
        -e "s|@CONSD_PORT@|${CONSD_PORT}|g" \
        -e "s|@OLLAMA_PORT@|${OLLAMA_PORT}|g" \
        -e "s|@EMBED_PORT@|${EMBED_PORT}|g" \
        "$tpl" > "$out"
}

# psql helper that uses the bootstrap's connection params
pg() {
    PGPASSWORD="$DB_PASSWORD" psql \
        -h "$HOST_IP" -p "$DB_PORT" \
        -U "$DB_USER" -d "$DB_NAME" \
        -v ON_ERROR_STOP=1 "$@"
}

# Check sudo upfront so we don't get stuck mid-run
need_sudo() {
    if ! sudo -n true 2>/dev/null; then
        log "Several steps need sudo. Enter your password once now:"
        sudo -v || die "sudo required"
        # keep alive in background until script ends
        ( while true; do sudo -n true; sleep 50; kill -0 "$$" 2>/dev/null || exit; done ) &
    fi
}
