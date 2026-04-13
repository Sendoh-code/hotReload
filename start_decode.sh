#!/bin/bash
# 启动 vLLM 实例 1 (port 8001)
# 与实例 0 完全对等，用于 fault tolerance。
# 单卡运行时限制显存占用为 40%，与实例 0 各占一半。

source ~/LMCache_minIO/LMCache/.venv/bin/activate
unset LD_LIBRARY_PATH
export AWS_ACCESS_KEY_ID="minioadmin"
export AWS_SECRET_ACCESS_KEY="minioadmin"
export AWS_ENDPOINT_URL="http://localhost:9000"
export AWS_DEFAULT_REGION="us-east-1"
export PYTHONHASHSEED=0
export PYTHONPATH=/home/ext_yiti6755_colorado_edu/LMCache_minIO${PYTHONPATH:+:$PYTHONPATH}
export LD_LIBRARY_PATH=$(python -c "import torch; import os; print(os.path.dirname(torch.__file__))")/lib
export LMCACHE_CONFIG_FILE=/home/ext_yiti6755_colorado_edu/LMCache_minIO/lmcache_decode.yaml

python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-0.5B \
  --port 8001 \
  --gpu-memory-utilization 0.4 \
  --enable-chunked-prefill \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both","kv_connector_extra_config":{"lmcache_config_file":"/home/ext_yiti6755_colorado_edu/LMCache_minIO/lmcache_decode.yaml"}}'
