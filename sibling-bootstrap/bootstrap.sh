#!/usr/bin/env bash
# bootstrap.sh — turnkey new-VM setup for the drift-memory cognitive stack.
#
# Usage:
#   ./bootstrap.sh <agent-name>       # e.g. ./bootstrap.sh nova
#   ./bootstrap.sh <agent-name> --skip system,consd      # skip phases
#
# Environment overrides (advanced):
#   HOST_IP, DB_PORT, DB_USER, DB_PASSWORD,
#   MEMORY_REPO, LEGACY_CONSD_TYPO=1
#
# Idempotent — safe to re-run. Each phase detects already-installed state.

set -euo pipefail

# ---------- arg parse ----------
if [ $# -lt 1 ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    cat <<USAGE
Usage: $0 <agent-name> [--skip phase1,phase2,...]

Phases (in order):
  system    apt + ghostty repo + VBox shared folder mounts
  user      ~/.local/bin tools (uv, zellij, zig, yazi, bun, deno, tigerbeetle)
  claude    Claude Code (native)
  memory    drift-memory clone + .env + Postgres schema for <agent-name>
  hooks     Claude Code hook wrapper + settings.json + statusline
  tunnels   socat tunnels + @reboot crontab
  consd     Consolidation daemon docker container

Example:
  $0 nova
  $0 nova --skip system,user      # only refresh memory/hooks/tunnels
USAGE
    exit 0
fi

export AGENT_NAME="$1"; shift

if ! [[ "$AGENT_NAME" =~ ^[a-z][a-z0-9_]{1,30}$ ]]; then
    echo "ERROR: agent-name must match ^[a-z][a-z0-9_]{1,30}\$ (got: $AGENT_NAME)" >&2
    exit 1
fi

SKIP=""
while [ $# -gt 0 ]; do
    case "$1" in
        --skip) SKIP="$2"; shift 2 ;;
        --skip=*) SKIP="${1#*=}"; shift ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

skipped() {
    case ",$SKIP," in *",$1,"*) return 0 ;; *) return 1 ;; esac
}

# ---------- locate self & source common ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PKG_DIR_OVERRIDE="$SCRIPT_DIR"  # for use before _common.sh runs
# shellcheck source=lib/_common.sh
source "$SCRIPT_DIR/lib/_common.sh"
# Override PKG_DIR / TEMPLATES_DIR if running from non-default location
PKG_DIR="$SCRIPT_DIR"
TEMPLATES_DIR="$PKG_DIR/templates"

# ---------- summary + confirm ----------
cat <<SUMMARY

${C_BLU}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${C_RST}
  Sibling Bootstrap — agent: ${C_GRN}${AGENT_NAME}${C_RST}
${C_BLU}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${C_RST}

  Memory dir:  $MEMORY_DIR
  DB target:   $DB_USER@$HOST_IP:$DB_PORT/$DB_NAME (schema=$AGENT_NAME)
  Memory repo: $MEMORY_REPO
  Skip:        ${SKIP:-<none>}

SUMMARY

read -r -p "  Proceed? [y/N] " yn
case "$yn" in [yY]*) : ;; *) echo "aborted"; exit 0 ;; esac

# ---------- run phases ----------
PHASES=(
    "system:lib/01_system.sh"
    "user:lib/02_userland.sh"
    "claude:lib/03_claude.sh"
    "memory:lib/04_memory.sh"
    "hooks:lib/05_hooks.sh"
    "tunnels:lib/06_tunnels.sh"
    "consd:lib/07_consd.sh"
)

for entry in "${PHASES[@]}"; do
    name="${entry%%:*}"
    file="${entry#*:}"
    if skipped "$name"; then
        warn "Skipping phase: $name"
        continue
    fi
    if ! [ -f "$SCRIPT_DIR/$file" ]; then
        err "Missing phase script: $file"; exit 1
    fi
    # shellcheck source=/dev/null
    source "$SCRIPT_DIR/$file"
done

# ---------- final summary ----------
cat <<DONE

${C_GRN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${C_RST}
${C_GRN}  Bootstrap complete for: ${AGENT_NAME}${C_RST}
${C_GRN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${C_RST}

Next steps:

  1. ${C_YEL}Reboot${C_RST} if VBox shared folders aren't mounted yet (check above).
  2. Open a new shell so PATH and group memberships (vboxsf, docker) refresh.
  3. ${C_YEL}cd $MEMORY_DIR && claude${C_RST}
  4. Paste the contents of ${C_BLU}$PKG_DIR/BOOTSTRAP_PROMPT.md${C_RST} into the
     first Claude session — it will:
       • verify substrate health
       • write its own personalized CLAUDE.md identity doc
       • generate first Merkle attestation
       • store its first memory

  5. After that, every future session wakes with full continuity.

  Quick sanity check:
    ${C_DIM}$AGENT_NAME toolkit.py health${C_RST}
    ${C_DIM}ss -tlnp | grep -E ':(8082|8083|11434)'${C_RST}
    ${C_DIM}crontab -l${C_RST}

DONE
