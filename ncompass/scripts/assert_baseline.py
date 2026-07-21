#!/usr/bin/env python3
"""Parse a vllm-bench output file and assert it reproduces the baseline within tolerance.

Anchors all metrics to the main "Serving Benchmark Result" block (before the Steady-State
section), and requires the full request count / concurrency so a partial or lighter run
cannot masquerade as a valid baseline.
"""
import re, sys

THROUGHPUT_MIN     = 6680.0   # tok/s   (baseline 6783 ±16, ~1.5% band)
TPOT_MED_MAX       = 86.3     # ms      (baseline 85.44, ~1% band)
TTFT_MED_MAX       = 320.0    # ms      (baseline ~259, generous band)
EXPECT_REQUESTS    = 6000
EXPECT_CONCURRENCY = 600


def num(pat, text):
    m = re.search(pat, text)
    if not m:
        sys.exit(f"metric not found: {pat}")
    return float(m.group(1))


def main(path):
    full = open(path).read()
    # anchor to the main result block; drop any Steady-State / windowed section
    t = re.split(r"Steady-State", full)[0]
    thr     = num(r"Output token throughput \(tok/s\):\s*([\d.]+)", t)
    tpot    = num(r"Median TPOT \(ms\):\s*([\d.]+)", t)
    ttft    = num(r"Median TTFT \(ms\):\s*([\d.]+)", t)
    failed  = num(r"Failed requests:\s*([\d.]+)", t)
    success = num(r"Successful requests:\s*([\d.]+)", t)
    conc    = num(r"Maximum request concurrency:\s*([\d.]+)", t)
    ok = (thr >= THROUGHPUT_MIN and tpot <= TPOT_MED_MAX and ttft <= TTFT_MED_MAX
          and failed == 0 and success == EXPECT_REQUESTS and conc == EXPECT_CONCURRENCY)
    print(f"throughput={thr} tpot_med={tpot} ttft_med={ttft} failed={failed} "
          f"success={success} conc={conc} -> {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main(sys.argv[1])
