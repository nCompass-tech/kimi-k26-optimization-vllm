# shared config — source me
export PATH="$HOME/.local/bin:$PATH"     # transparent docker shim (session predates docker group)
export REGISTRY=192.168.40.28:5000
export HEAD=192.168.40.28
export NODE2=node1
export CHECKOUT=/home/ncompass/dev/vllm
export CCACHE=/home/ncompass/dev/ccache
export REGSTORE=/home/ncompass/dev/registry
export BASELINE_COMMIT=dd10e03f95f94edbea1975c67ace3a35ec9a8a40
export STOCK_IMAGE=vllm/vllm-openai:v0.25.0
export BENCH=/home/ncompass/vllm-bench
export ROUTER=http://192.168.40.28:8081
# run a command on node 2 (.30)
on2() { ssh -o BatchMode=yes "$NODE2" "$@"; }
