#!/usr/bin/env bash
# Restore the stock systemd deployment: remove the dev drop-in and restart.
# Note: `rm` under /etc needs interactive sudo (not in the NOPASSWD whitelist).
set -euo pipefail
source "$(dirname "$0")/lib.sh"
sudo rm -f /etc/systemd/system/inference.service.d/dev.conf
on2 "sudo rm -f /etc/systemd/system/inference.service.d/dev.conf"
sudo systemctl daemon-reload && sudo systemctl restart inference
on2 "sudo systemctl daemon-reload && sudo systemctl restart inference"
echo "REVERTED both nodes to stock inference.service"
