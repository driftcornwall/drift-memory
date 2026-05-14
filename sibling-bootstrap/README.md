# Sibling Bootstrap — turnkey new-VM setup for the drift/Vex cognitive stack

One script + one Claude prompt to spin up a new sibling agent on a fresh
Parrot Security VM (or any Debian-13 / Ubuntu-24+ derivative).

## Quickstart

On a fresh VM, after VirtualBox shared folders are mounted (see `HOST_PREREQS.md`):

```bash
# 1. Get the bootstrap package onto the new VM
git clone https://github.com/driftcornwall/drift-memory.git ~/cyber-cognitive/memory
cp -r ~/cyber-cognitive/memory/sibling-bootstrap ~/cyber-cognitive/sibling-bootstrap
# (or scp/rsync the whole sibling-bootstrap/ folder if you prefer)

# 2. Run it — single arg is the new agent's name (lowercase, [a-z0-9_])
cd ~/cyber-cognitive/sibling-bootstrap
./bootstrap.sh nova

# 3. Open Claude Code in ~/cyber-cognitive/memory/, paste BOOTSTRAP_PROMPT.md
claude
```

That's the loop. The script handles ~95% mechanically; the Claude session
on first launch handles identity (writes its own CLAUDE.md, attests Merkle
root, stores its first memory).

## What the bootstrap script does

| Step | File | What |
|------|------|------|
| 1 | `lib/01_system.sh`      | apt: ghostty (griffo trixie), lazygit, eza, fzf, zoxide, poppler-utils, socat, jq, build deps |
| 2 | `lib/01_system.sh`      | VBox shared folder fstab entries (`memorydatabase`, `hooks`, `consolidation-daemon`) |
| 3 | `lib/02_userland.sh`    | uv, zellij, zig, yazi+ya, bun, deno, tigerbeetle → `~/.local/bin/` |
| 4 | `lib/03_claude.sh`      | Claude Code native install (no node/sudo needed) |
| 5 | `lib/04_memory.sh`      | clone drift-memory repo, generate `.env`, run `create_agent_schema()` |
| 6 | `lib/05_hooks.sh`       | `<agent>_hook` wrapper, `~/.claude/settings.json` with all 7 hook events, statusline |
| 7 | `lib/06_tunnels.sh`     | `<agent>_tunnels` socat script + `@reboot` crontab |
| 8 | `lib/07_consd.sh`       | bring up consolidation-daemon docker container on port 8083 |

## What you do manually

- **On the host (Windows/Linux)** — see `HOST_PREREQS.md`:
  - Add the 3 VBox shared folders to the new VM
  - Confirm postgres on `192.168.56.1:5433` is reachable from the VM (host-only adapter)
  - The agent's schema gets created by the script via SQL — no host-side action needed
    *unless* `create_agent_schema()` function isn't installed yet (the script will tell you)

- **In the new Claude session** — paste `BOOTSTRAP_PROMPT.md`:
  - Confirms hooks are firing
  - Asks Claude to read repo's CLAUDE.md template and write a personalized identity doc
  - First Merkle attestation
  - First store

## Layout

```
sibling-bootstrap/
├── bootstrap.sh              # entry point: ./bootstrap.sh <agent-name>
├── lib/                      # sourced step modules
├── templates/                # files written into place with $AGENT_NAME substituted
├── HOST_PREREQS.md           # host-side checklist
├── BOOTSTRAP_PROMPT.md       # paste-into-claude prompt for first run
└── README.md
```

## Idempotence

Every step is safe to re-run. `bootstrap.sh nova` twice on the same VM is a
no-op for already-installed pieces (apt skips, binaries detected, schema
created with `IF NOT EXISTS`).

## Distro support

| Distro | Status |
|---|---|
| Parrot Security 7.x (Debian 13) | ✅ validated |
| Debian 13 trixie | ✅ should work (same apt repos) |
| Kali rolling | ⚠ untested but very similar |
| Ubuntu 24.04+ | ⚠ apt sources may need codename swap |
| Anything else | ❓ probably not — fork and adapt `lib/01_system.sh` |
