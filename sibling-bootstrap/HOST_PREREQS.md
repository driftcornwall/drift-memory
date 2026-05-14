# Host Prerequisites

Things that must be true on the host machine (not the VM) before
`bootstrap.sh` can succeed on the VM.

## 1. VirtualBox shared folders

Add these three shared folders to the new VM in VirtualBox:

| Host path (your choice) | Folder name (must match) | Auto-mount | Make permanent |
|---|---|---|---|
| `Q:/.../memorydatabase`         | `memorydatabase`        | ✅ | ✅ |
| `Q:/.../hooks`                  | `hooks`                 | ✅ | ✅ |
| `Q:/.../consolidation-daemon`   | `consolidation-daemon`  | ✅ | ✅ |

The **folder name** column is the contract — bootstrap.sh writes fstab
entries that reference exactly those names. Host paths don't matter.

> **Note on the typo:** the original Vex VM mounts to `~/consolidation-deamon`
> (missing 'a'). The new bootstrap fixes this to `~/consolidation-daemon`.
> If you're cloning state from Vex and it relies on the typo, set
> `LEGACY_CONSD_TYPO=1` before running bootstrap.sh.

## 2. Postgres reachable from VM

The cognitive stack depends on a Postgres on the host with pgvector,
exposing port `5433` on the **host-only adapter** (e.g. `192.168.56.1`).

Confirm from the VM:

```bash
nc -zv 192.168.56.1 5433
```

Credentials in your current Vex `.env`:
- DB: `agent_memory`
- User: `agent_admin`
- Password: `agent_memory_local_dev`  (change for production)

If the new agent's schema doesn't exist, bootstrap.sh runs:
```sql
SELECT create_agent_schema('<new-agent-name>');
```
That function comes from `~/cyber-cognitive/memory/docs/schema.sql`. If
`create_agent_schema` isn't installed in your DB yet, run the schema.sql
file once on the host:

```bash
psql -h 192.168.56.1 -p 5433 -U agent_admin -d agent_memory \
     -f ~/cyber-cognitive/memory/docs/schema.sql
```

## 3. Supporting services on the host

The bootstrap script does NOT install these — they live on the host and
serve all sibling VMs. Check each is up and listening on the host-only
adapter (`192.168.56.1`):

| Service | Port | Used for |
|---|---|---|
| Postgres + pgvector  | 5433  | memory store |
| Embedding service    | 8080  | semantic search vectors |
| NLI service          | 8082  | contradiction detection |
| Ollama               | 11434 | gemma_bridge vocabulary expansion |

Quick host-check from the new VM after VBox network is up:

```bash
for p in 5433 8080 8082 11434; do
    timeout 2 bash -c "</dev/tcp/192.168.56.1/$p" 2>/dev/null \
        && echo "  $p OPEN" || echo "  $p CLOSED"
done
```

## 4. VirtualBox Guest Additions on the VM

For VBox shared folders to mount, the guest VM must have Guest Additions
installed. On Parrot/Debian:

```bash
sudo apt install -y virtualbox-guest-utils virtualbox-guest-x11
sudo usermod -aG vboxsf $USER     # then log out and back in
```

Then reboot — fstab entries written by bootstrap.sh need GA loaded to mount.
