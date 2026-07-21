#!/usr/bin/env bash
# health.sh [--wait] [expected-image]
#   Checks engines :8000 on both nodes + router :8081. If expected-image is given, asserts each
#   container's image ID equals the pulled digest of that image (defeats stale mutable tags).
set -uo pipefail
source "$(dirname "$0")/lib.sh"
WAIT=0; [ "${1:-}" = "--wait" ] && { WAIT=1200; shift; }
EXPECT=${1:-}
deadline=$(( $(date +%s) + WAIT ))

want_id=""
if [ -n "$EXPECT" ]; then
  want_id=$(docker inspect --format '{{.Id}}' "$EXPECT" 2>/dev/null || true)
fi
check() {
  curl -fsS -m 5 http://192.168.40.28:8000/health >/dev/null 2>&1 && e28=1 || e28=0
  on2 'curl -fsS -m 5 http://192.168.40.30:8000/health >/dev/null 2>&1' && e30=1 || e30=0
  curl -fsS -m 5 http://192.168.40.28:8081/health >/dev/null 2>&1 && rtr=1 || rtr=0
  [ "$e28" = 1 ] && [ "$e30" = 1 ] && [ "$rtr" = 1 ]
}
while :; do
  if check; then
    i28=$(docker inspect --format '{{.Config.Image}}' inference 2>/dev/null)
    i30=$(on2 "docker inspect --format '{{.Config.Image}}' inference 2>/dev/null")
    if [ -n "$want_id" ]; then
      # compare the running container's image ID to the pulled digest on each node
      r28=$(docker inspect --format '{{.Image}}' inference 2>/dev/null)
      r30=$(on2 "docker inspect --format '{{.Image}}' inference 2>/dev/null")
      w30=$(on2 "docker inspect --format '{{.Id}}' '$EXPECT' 2>/dev/null")
      if [ "$r28" != "$want_id" ] || [ "$r30" != "$w30" ]; then
        echo "IMAGE MISMATCH: want28=$want_id run28=$r28 ; want30=$w30 run30=$r30"; exit 2
      fi
    fi
    echo "HEALTHY  engine28=$i28  engine30=$i30  router=up"; exit 0
  fi
  [ "$(date +%s)" -ge "$deadline" ] && { echo "UNHEALTHY e28=$e28 e30=$e30 rtr=$rtr"; exit 1; }
  sleep 10
done
