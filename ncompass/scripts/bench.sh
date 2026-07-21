#!/usr/bin/env bash
# usage: bench.sh <label>   1 warm-up + 2 measured full runs; asserts each vs baseline.
set -euo pipefail
source "$(dirname "$0")/lib.sh"
LABEL=${1:?label required}
cd "$(dirname "$0")/.."
mkdir -p runs
BENCH_ARGS=(--backend openai-chat --model moonshotai/Kimi-K2.6 --base-url "$ROUTER" \
  --max-concurrency 600 --num-prompts 6000 --dataset-name random \
  --random-input-len 5120 --random-output-len 1280 --random-range-ratio 0.5 \
  --random-prefix-len 4096 --ignore-eos --trust-remote-code)
echo "== warm-up (not measured) =="
"$BENCH" "${BENCH_ARGS[@]}" --num-prompts 600 >/dev/null 2>&1 || true
rc=0
for i in 1 2; do
  echo "== measured run $i =="
  # write to file directly (no tee-in-pipeline, which would trip pipefail if bench warn-exits),
  # tolerate a non-zero bench exit so both runs + both asserts always execute
  "$BENCH" "${BENCH_ARGS[@]}" > "runs/$LABEL-run$i.txt" 2>&1 || echo "(bench exited $? — see runs/$LABEL-run$i.txt)"
  tail -30 "runs/$LABEL-run$i.txt"
  python3 scripts/assert_baseline.py "runs/$LABEL-run$i.txt" || rc=1
done
exit $rc
