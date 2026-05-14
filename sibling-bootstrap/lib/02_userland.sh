# shellcheck shell=bash
# User-space binaries into ~/.local/bin.

step "03 — Userland binaries (~/.local/bin)"

ensure_dir "$LOCAL_BIN"
case ":$PATH:" in
    *":$LOCAL_BIN:"*) : ;;
    *) warn "$LOCAL_BIN not in PATH — add to ~/.bashrc: export PATH=\"$LOCAL_BIN:\$PATH\"" ;;
esac

# ---- uv (must be first; other installs lean on it indirectly) ----
if ! [ -x "$LOCAL_BIN/uv" ]; then
    log "Installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh >/tmp/uv-install.log 2>&1 || warn "uv install had errors — see /tmp/uv-install.log"
fi
have uv && ok "uv $(uv --version | head -1)"

# ---- zellij (terminal multiplexer) ----
if ! [ -x "$LOCAL_BIN/zellij" ]; then
    log "Installing zellij"
    Z_URL=$(curl -sL https://api.github.com/repos/zellij-org/zellij/releases/latest \
        | python3 -c "import json,sys;d=json.load(sys.stdin);print([a['browser_download_url'] for a in d['assets'] if 'x86_64-unknown-linux-musl.tar.gz' in a['name']][0])")
    curl -fsSL "$Z_URL" -o /tmp/zellij.tgz
    tar -xzf /tmp/zellij.tgz -C "$LOCAL_BIN" zellij
    chmod +x "$LOCAL_BIN/zellij"
    rm -f /tmp/zellij.tgz
fi
ok "zellij $($LOCAL_BIN/zellij --version 2>/dev/null | head -1)"

# ---- zig (latest stable) ----
if ! [ -x "$LOCAL_BIN/zig" ]; then
    log "Installing zig (latest stable)"
    Z_INFO=$(curl -s https://ziglang.org/download/index.json \
        | python3 -c "import json,sys;d=json.load(sys.stdin);k=[x for x in d if x!='master'][0];print(k);print(d[k]['x86_64-linux']['tarball'])")
    Z_VER=$(echo "$Z_INFO" | head -1)
    Z_URL=$(echo "$Z_INFO" | tail -1)
    curl -fsSL "$Z_URL" -o /tmp/zig.tar.xz
    mkdir -p "$AGENT_HOME/.local/zig"
    tar -xf /tmp/zig.tar.xz -C "$AGENT_HOME/.local/zig"
    rm -rf "$AGENT_HOME/.local/zig/$Z_VER"
    mv "$AGENT_HOME/.local/zig/zig-x86_64-linux-$Z_VER" "$AGENT_HOME/.local/zig/$Z_VER"
    ln -sf "$AGENT_HOME/.local/zig/$Z_VER/zig" "$LOCAL_BIN/zig"
    rm -f /tmp/zig.tar.xz
fi
ok "zig $($LOCAL_BIN/zig version 2>/dev/null)"

# ---- yazi + ya ----
if ! [ -x "$LOCAL_BIN/yazi" ]; then
    log "Installing yazi"
    Y_URL=$(curl -sL https://api.github.com/repos/sxyazi/yazi/releases/latest \
        | python3 -c "import json,sys;d=json.load(sys.stdin);print([a['browser_download_url'] for a in d['assets'] if 'x86_64-unknown-linux-gnu.zip' in a['name']][0])")
    curl -fsSL "$Y_URL" -o /tmp/yazi.zip
    unzip -q -o /tmp/yazi.zip -d /tmp/yazi-extract
    install -m755 /tmp/yazi-extract/*/yazi "$LOCAL_BIN/yazi"
    install -m755 /tmp/yazi-extract/*/ya   "$LOCAL_BIN/ya"
    rm -rf /tmp/yazi.zip /tmp/yazi-extract
fi
ok "yazi $($LOCAL_BIN/yazi --version 2>/dev/null | head -1)"

# ---- bun ----
if ! [ -x "$LOCAL_BIN/bun" ]; then
    log "Installing bun"
    BUN_INSTALL="$AGENT_HOME/.bun" curl -fsSL https://bun.sh/install \
        | bash >/tmp/bun-install.log 2>&1 || warn "bun install had errors"
    ln -sf "$AGENT_HOME/.bun/bin/bun"  "$LOCAL_BIN/bun"
    ln -sf "$AGENT_HOME/.bun/bin/bunx" "$LOCAL_BIN/bunx"
fi
ok "bun $($LOCAL_BIN/bun --version 2>/dev/null)"

# ---- deno ----
if ! [ -x "$LOCAL_BIN/deno" ]; then
    log "Installing deno"
    DENO_INSTALL="$AGENT_HOME/.deno" curl -fsSL https://deno.land/install.sh \
        | sh -s -- -y >/tmp/deno-install.log 2>&1 || warn "deno install had errors"
    ln -sf "$AGENT_HOME/.deno/bin/deno" "$LOCAL_BIN/deno"
fi
ok "deno $($LOCAL_BIN/deno --version 2>/dev/null | head -1)"

# ---- tigerbeetle ----
if ! [ -x "$LOCAL_BIN/tigerbeetle" ]; then
    log "Installing tigerbeetle"
    curl -fsSL -o /tmp/tigerbeetle.zip https://linux.tigerbeetle.com
    unzip -q -o /tmp/tigerbeetle.zip -d /tmp/tb-extract
    install -m755 /tmp/tb-extract/tigerbeetle "$LOCAL_BIN/tigerbeetle"
    rm -rf /tmp/tigerbeetle.zip /tmp/tb-extract
fi
ok "tigerbeetle $($LOCAL_BIN/tigerbeetle version 2>/dev/null | head -1)"
