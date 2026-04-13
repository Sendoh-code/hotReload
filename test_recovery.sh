#!/bin/bash
# 验证 hot cache 预热恢复的端到端测试
# 用法: ./test_recovery.sh
#
# 前提: MinIO / LMCache Controller / GMS 已在运行
# 测试流程:
#   1. 启动两个 vLLM 实例
#   2. 发送推理请求，填充 cache
#   3. Kill 实例 0，记录无 cache 时的 TTFT（对照组）
#   4. 重启实例 0，等待预热完成
#   5. 发送相同请求，记录有 cache 时的 TTFT（实验组）

VLLM0_PORT=8000
VLLM1_PORT=8001
MODEL="Qwen/Qwen2.5-0.5B"
LOG_DIR="/tmp/lmcache_test_logs"
mkdir -p "$LOG_DIR"

# 用于测试的固定 prompt（足够长以产生多个 chunk）
PROMPT="Once upon a time in a land far away, there lived a wise old wizard who had spent centuries studying the ancient arts of magic. He had accumulated vast knowledge about the universe, the stars, and the hidden forces that governed all living things. His tower stood at the edge of the kingdom, visible from every corner of the realm."

# ─── 辅助函数 ─────────────────────────────────────────────────────────────────

wait_for_vllm() {
    local port=$1
    local max_wait=120
    local elapsed=0
    echo "等待 vLLM port $port 就绪..."
    while ! curl -s "http://localhost:$port/health" > /dev/null 2>&1; do
        sleep 2
        elapsed=$((elapsed + 2))
        if [ $elapsed -ge $max_wait ]; then
            echo "ERROR: vLLM port $port 启动超时"
            exit 1
        fi
    done
    echo "vLLM port $port 已就绪"
}

send_request_timed() {
    local port=$1
    local label=$2
    local start end elapsed_ms ttft

    start=$(date +%s%3N)
    response=$(curl -s -X POST "http://localhost:$port/v1/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"$MODEL\",
            \"prompt\": \"$PROMPT\",
            \"max_tokens\": 50,
            \"temperature\": 0
        }")
    end=$(date +%s%3N)
    elapsed_ms=$((end - start))

    # 提取生成内容
    generated=$(echo "$response" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['text'][:80])" 2>/dev/null || echo "(parse error)")

    echo "[$label] 耗时=${elapsed_ms}ms  output=${generated}"
    echo "$elapsed_ms"
}

# ─── Step 1: 启动 vLLM 实例 ──────────────────────────────────────────────────

echo ""
echo "=== Step 1: 启动两个 vLLM 实例 ==="

bash ~/LMCache_minIO/start_prefill.sh > "$LOG_DIR/vllm0.log" 2>&1 &
VLLM0_PID=$!
echo "实例 0 PID=$VLLM0_PID  log=$LOG_DIR/vllm0.log"

bash ~/LMCache_minIO/start_decode.sh > "$LOG_DIR/vllm1.log" 2>&1 &
VLLM1_PID=$!
echo "实例 1 PID=$VLLM1_PID  log=$LOG_DIR/vllm1.log"

wait_for_vllm $VLLM0_PORT
wait_for_vllm $VLLM1_PORT

# ─── Step 2: 填充 cache ──────────────────────────────────────────────────────

echo ""
echo "=== Step 2: 发送推理请求填充 cache ==="

echo "首次请求（冷启动，会写入 MinIO）..."
send_request_timed $VLLM0_PORT "首次请求-实例0" > /dev/null
send_request_timed $VLLM0_PORT "首次请求-实例0-确认" > /dev/null

# 等待 GMS commit_write 完成（异步写入）
echo "等待 3 秒让 GMS commit_write 完成..."
sleep 3

# 验证 GMS 已记录 blocks
echo "GMS 当前 blocks:"
curl -s "http://localhost:8500/blocks" 2>/dev/null || \
    python3 -c "
import socket, json
s = socket.socket()
s.connect(('localhost', 8500))
s.sendall(json.dumps({'id':'x','method':'list_blocks','params':{'limit':10}}).encode() + b'\n')
resp = s.makefile('r').readline()
d = json.loads(resp)
print(f'  committed blocks: {len(d.get(\"blocks\",[]))}')
s.close()
"

# ─── Step 3: Kill 实例 0，测量无 cache 的冷启动 TTFT（对照组）─────────────────

echo ""
echo "=== Step 3: Kill 实例 0 ==="
kill $VLLM0_PID
sleep 2
# 确保进程已死
kill -9 $VLLM0_PID 2>/dev/null

echo "实例 0 已 kill。等待 15 秒让 GMS 检测到 failure..."
sleep 15

# 重启实例 0（不等待预热，立即测对照组）
echo ""
echo "=== Step 4a: 重启实例 0（对照组：不等预热完成）==="
bash ~/LMCache_minIO/start_prefill.sh > "$LOG_DIR/vllm0_restart.log" 2>&1 &
VLLM0_NEW_PID=$!
wait_for_vllm $VLLM0_PORT

echo "立即发送相同请求（CPU cache 尚未预热，走 MinIO）..."
COLD_MS=$(send_request_timed $VLLM0_PORT "对照组-重启后立即")
echo "对照组 TTFT = ${COLD_MS}ms"

# 等待后台预热线程完成
echo ""
echo "=== Step 4b: 等待预热线程完成 ==="
echo "检查日志中的预热完成信息（最多等 60 秒）..."
elapsed=0
while ! grep -q "GMS recovery: done" "$LOG_DIR/vllm0_restart.log" 2>/dev/null; do
    sleep 2
    elapsed=$((elapsed + 2))
    if [ $elapsed -ge 60 ]; then
        echo "WARNING: 超时未见预热完成日志，继续测试..."
        break
    fi
done

grep "GMS recovery" "$LOG_DIR/vllm0_restart.log" | tail -5

# ─── Step 5: 测量预热后的 TTFT（实验组）─────────────────────────────────────

echo ""
echo "=== Step 5: 测量预热后 TTFT（实验组：CPU cache 已预热）==="
WARM_MS=$(send_request_timed $VLLM0_PORT "实验组-预热后")
echo "实验组 TTFT = ${WARM_MS}ms"

# ─── 结果对比 ────────────────────────────────────────────────────────────────

echo ""
echo "=== 结果对比 ==="
echo "  对照组（重启后立即，走 MinIO）: ${COLD_MS}ms"
echo "  实验组（预热完成后，走 CPU）:   ${WARM_MS}ms"

if [ -n "$COLD_MS" ] && [ -n "$WARM_MS" ] && [ "$WARM_MS" -lt "$COLD_MS" ]; then
    speedup=$(echo "scale=1; ($COLD_MS - $WARM_MS) * 100 / $COLD_MS" | bc)
    echo "  预热提升: ${speedup}%"
else
    echo "  无明显提升或数据异常，请查看日志"
fi

echo ""
echo "日志位置: $LOG_DIR/"
echo "  vllm0_restart.log 中搜索 'GMS recovery' 查看预热详情"

# 清理
kill $VLLM0_NEW_PID $VLLM1_PID 2>/dev/null
