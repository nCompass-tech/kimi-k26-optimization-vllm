#!/usr/bin/env bash
# One-time node bootstrap for the Kimi decode-opt dev flow.
# Run as root on EACH node:  sudo bash bootstrap-node.sh
# Idempotent: safe to re-run.
set -euo pipefail

# 1) Bring the stock engine down (Restart=no keeps it down until we redeploy).
systemctl stop inference 2>/dev/null || true

# 2) Give ncompass docker access (append; harmless if already a member).
usermod -aG docker ncompass

# 3) NOPASSWD sudoers for exactly the systemctl verbs the deploy loop needs.
#    Explicit verbs only — NOT a `* inference` wildcard, which would also match
#    `systemctl status inference` (spawns a pager as root → `!sh` = root shell).
cat > /etc/sudoers.d/ncompass-inference <<'SUDO'
ncompass ALL=(root) NOPASSWD: /usr/bin/systemctl daemon-reload
ncompass ALL=(root) NOPASSWD: /usr/bin/systemctl restart inference
ncompass ALL=(root) NOPASSWD: /usr/bin/systemctl stop inference
ncompass ALL=(root) NOPASSWD: /usr/bin/systemctl start inference
ncompass ALL=(root) NOPASSWD: /usr/bin/systemctl is-active inference
SUDO
chmod 440 /etc/sudoers.d/ncompass-inference

# 4) Validate the sudoers file (fails loudly if malformed).
visudo -cf /etc/sudoers.d/ncompass-inference

echo "BOOTSTRAP_OK on $(hostname): engine stopped, ncompass in docker group, sudoers valid"
