#!/usr/bin/env bash
# Verify the environment is ready for the dev-flow pipeline.
set -uo pipefail
source "$(dirname "$0")/lib.sh"
fail=0
echo "[.28] docker usable:";  docker version >/dev/null 2>&1 && echo "  ok" || { echo "  NO ACCESS"; fail=1; }
echo "[.30] ssh + docker:";   on2 'docker version >/dev/null 2>&1' && echo "  ok" || { echo "  FAIL"; fail=1; }
echo "[.28] NOPASSWD systemctl:"; sudo -n systemctl daemon-reload >/dev/null 2>&1 && echo "  ok" || { echo "  needs password"; fail=1; }
echo "[.30] NOPASSWD systemctl:"; on2 'sudo -n systemctl daemon-reload >/dev/null 2>&1' && echo "  ok" || { echo "  needs password"; fail=1; }
echo "[.28] registry :5000:"; curl -fsS -m5 "http://$HEAD:5000/v2/" >/dev/null 2>&1 && echo "  ok" || { echo "  DOWN"; fail=1; }
echo "[.30] sees registry:";  on2 "curl -fsS -m5 http://$HEAD:5000/v2/ >/dev/null 2>&1" && echo "  ok" || { echo "  UNREACHABLE"; fail=1; }
echo "dirs:"; for d in "$CHECKOUT" "$CCACHE" "$REGSTORE"; do [ -d "$d" ] && echo "  $d ok" || { echo "  $d MISSING"; fail=1; }; done
echo "[.28] engine down:"; curl -fsS -m3 http://192.168.40.28:8000/health >/dev/null 2>&1 && { echo "  UP(!)"; fail=1; } || echo "  down ok"
echo "[.30] engine down:"; on2 'curl -fsS -m3 http://192.168.40.30:8000/health >/dev/null 2>&1' && { echo "  UP(!)"; fail=1; } || echo "  down ok"
[ $fail -eq 0 ] && echo "PREFLIGHT_OK" || echo "PREFLIGHT_FAIL"
exit $fail
