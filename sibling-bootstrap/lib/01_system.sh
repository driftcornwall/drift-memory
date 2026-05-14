# shellcheck shell=bash
# System packages, ghostty repo, VBox shared folder mounts.

step "01 — System packages"
need_sudo

# ---------- detect distro / codename ----------
. /etc/os-release
log "Detected: ${PRETTY_NAME:-unknown} (id=$ID, codename=${VERSION_CODENAME:-unknown})"

# ghostty griffo.io repo only ships for known Debian codenames.
# Parrot 7.x reports codename 'echo' but is built on Debian 13 'trixie'.
# Map known parrot codenames → underlying debian codename.
case "${VERSION_CODENAME:-}" in
    echo)        DEBIAN_CN=trixie  ;;   # Parrot 7.x
    lory|ara)    DEBIAN_CN=bookworm ;;  # earlier Parrot
    bookworm|trixie|forky|sid) DEBIAN_CN="$VERSION_CODENAME" ;;
    *)
        # Fallback: use /etc/debian_version major
        case "$(cat /etc/debian_version 2>/dev/null | cut -d. -f1)" in
            13) DEBIAN_CN=trixie ;;
            12) DEBIAN_CN=bookworm ;;
            *)  DEBIAN_CN=trixie ;;
        esac
        ;;
esac
log "Using griffo.io codename: $DEBIAN_CN"

# ---------- ghostty repo ----------
if ! [ -f /etc/apt/trusted.gpg.d/debian.griffo.io.gpg ]; then
    log "Installing griffo.io GPG key"
    curl -fsSL https://debian.griffo.io/EA0F721D231FDD3A0A17B9AC7808B4DD62C41256.asc \
        | sudo gpg --dearmor --yes -o /etc/apt/trusted.gpg.d/debian.griffo.io.gpg
fi

GRIFFO_LINE="deb https://debian.griffo.io/apt $DEBIAN_CN main"
if ! grep -qsxF "$GRIFFO_LINE" /etc/apt/sources.list.d/debian.griffo.io.list 2>/dev/null; then
    log "Writing griffo.io sources entry"
    echo "$GRIFFO_LINE" | sudo tee /etc/apt/sources.list.d/debian.griffo.io.list >/dev/null
fi

# ---------- apt update + install ----------
log "apt update"
sudo apt-get update -qq

APT_PKGS=(
    # ghostty
    ghostty
    # everyday CLI
    lazygit eza fzf zoxide
    # yazi previews
    poppler-utils ffmpeg fd-find ripgrep imagemagick
    # network/diagnostic
    socat netcat-openbsd jq
    # python (for hooks/uv)
    python3 python3-pip python3-venv
    # build deps in case anything compiles
    build-essential pkg-config libssl-dev
    # postgres client (for schema bootstrap)
    postgresql-client
    # docker for consolidation daemon
    docker.io docker-compose-v2
    # vbox guest if not installed
    virtualbox-guest-utils
)
log "apt install: ${APT_PKGS[*]}"
sudo apt-get install -y -qq "${APT_PKGS[@]}" || warn "some packages failed — continuing"

# Make sure user is in vboxsf + docker groups (takes effect on next login)
sudo usermod -aG vboxsf "$USER" 2>/dev/null || true
sudo usermod -aG docker "$USER" 2>/dev/null || true

ok "System packages installed"

step "02 — VBox shared folder mounts"

ensure_dir "${VBOX_MOUNT[memorydatabase]}"
ensure_dir "${VBOX_MOUNT[hooks]}"
ensure_dir "${VBOX_MOUNT[consolidation-daemon]}"

# Detect uid/gid (default Parrot puts users in gid 1002)
UID_NUM=$(id -u)
GID_NUM=$(id -g)

FSTAB_MARKER="# bootstrap.sh: VBox shared folders for drift-memory"
if ! grep -qF "$FSTAB_MARKER" /etc/fstab; then
    log "Adding fstab entries (uid=$UID_NUM gid=$GID_NUM)"
    {
        echo ""
        echo "$FSTAB_MARKER"
        for share in "${VBOX_SHARES[@]}"; do
            mnt="${VBOX_MOUNT[$share]}"
            printf "%-22s %s vboxsf uid=%s,gid=%s,nofail,_netdev 0 0\n" \
                "$share" "$mnt" "$UID_NUM" "$GID_NUM"
        done
    } | sudo tee -a /etc/fstab >/dev/null
    ok "fstab updated"
else
    ok "fstab entries already present"
fi

# Try to mount now (will fail silently if VBox GA module not loaded yet)
log "Attempting mount -a"
sudo mount -a 2>/dev/null || warn "mount -a had errors — likely needs reboot for vboxsf module"

for share in "${VBOX_SHARES[@]}"; do
    mnt="${VBOX_MOUNT[$share]}"
    if mountpoint -q "$mnt"; then
        ok "mounted: $share → $mnt"
    else
        warn "not mounted yet: $share → $mnt (reboot then re-run bootstrap.sh)"
    fi
done
