# LMCache + MinIO 热缓存容错恢复系统 技术报告

**日期**: 2026-04-13  
**环境**: 单机 NVIDIA L4 (23 GB)  
**模型**: Qwen/Qwen2.5-0.5B  

---

## 1. 项目背景与目标

### 1.1 核心问题

在多 vLLM 实例部署中，每个进程在内存中维护 KV cache（键值缓存）。当某个 vLLM 进程崩溃重启后，其内存中的 KV cache 全部丢失，导致重启后的首批请求必须从头重新计算 prefill，产生显著的 TTFT（Time To First Token）退化。

### 1.2 设计目标

- **容错性**：进程崩溃后，KV cache 数据持久化在对象存储中，不会永久丢失
- **热启动**：进程重启后，主动将 MinIO 中的 KV cache 预加载进 CPU 内存，使首批请求命中本地缓存而非远程对象存储
- **透明性**：上述逻辑完全在 LMCache 层完成，vLLM 主体代码无侵入式修改
- **可观测性**：通过 GMS 全局元数据服务追踪每个进程持有的 cache 块

---

## 2. 系统架构

### 2.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                       单机 (NVIDIA L4)                              │
│                                                                     │
│  ┌───────────────────────┐   ┌───────────────────────┐             │
│  │   vLLM Instance 0     │   │   vLLM Instance 1     │             │
│  │   (port 8000)         │   │   (port 8001)         │             │
│  │                       │   │                       │             │
│  │  ┌─────────────────┐  │   │  ┌─────────────────┐  │             │
│  │  │  LMCacheEngine  │  │   │  │  LMCacheEngine  │  │             │
│  │  │  (embedded lib) │  │   │  │  (embedded lib) │  │             │
│  │  │                 │  │   │  │                 │  │             │
│  │  │ ┌─────────────┐ │  │   │  │ ┌─────────────┐ │  │             │
│  │  │ │ CPU Cache   │ │  │   │  │ │ CPU Cache   │ │  │             │
│  │  │ │ (5 GB max)  │ │  │   │  │ │ (5 GB max)  │ │  │             │
│  │  │ └─────────────┘ │  │   │  │ └─────────────┘ │  │             │
│  │  │                 │  │   │  │                 │  │             │
│  │  │ ┌─────────────┐ │  │   │  │ ┌─────────────┐ │  │             │
│  │  │ │RemoteBackend│ │  │   │  │ │RemoteBackend│ │  │             │
│  │  │ │(S3 awscrt)  │ │  │   │  │ │(S3 awscrt)  │ │  │             │
│  │  │ └──────┬──────┘ │  │   │  │ └──────┬──────┘ │  │             │
│  │  │        │GMS RPC │  │   │  │        │GMS RPC │  │             │
│  │  └────────┼────────┘  │   │  └────────┼────────┘  │             │
│  └───────────┼───────────┘   └───────────┼───────────┘             │
│              │                           │                          │
│              └──────────┬────────────────┘                          │
│                         │ TCP + Line-JSON                           │
│                         ▼                                           │
│              ┌─────────────────────┐                               │
│              │  GMS Server         │                               │
│              │  (port 8500)        │                               │
│              │                     │                               │
│              │  Workers Registry   │                               │
│              │  Sequence Tracker   │                               │
│              │  Failover Engine    │                               │
│              └─────────────────────┘                               │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────┐     │
│  │  MinIO Object Storage (port 9000)                         │     │
│  │  Bucket: lmcache-bucket                                   │     │
│  │  Key format: {model}_{world}@{worker}@{hash}@{dtype}      │     │
│  │  Each block: 73,728 bytes (chunk_size=6 tokens)           │     │
│  └───────────────────────────────────────────────────────────┘     │
│                                                                     │
│  ┌───────────────────────┐                                         │
│  │  LMCache Controller   │                                         │
│  │  HTTP :8100           │                                         │
│  │  ZMQ Pull :8300       │                                         │
│  │  ZMQ Reply :8400      │                                         │
│  └───────────────────────┘                                         │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 两个 vLLM 实例的定位

与传统 PD 分离（Prefill-Decode 专职化）不同，本项目的两个实例**完全对等**，每个都启用 `--enable-chunked-prefill`，均可独立处理完整请求（prefill + decode）。

| 属性 | Instance 0 | Instance 1 |
|------|------------|------------|
| vLLM HTTP port | 8000 | 8001 |
| GPU memory util | 40% | 40% |
| LMCache config | lmcache_prefill.yaml | lmcache_decode.yaml |
| LMCache Worker ZMQ | :8050 | :8051 |
| GMS seq_id | `dev:8000` | `dev:8001` |
| 角色 | 对等节点 | 对等节点 |

> 选择对等架构（而非 PD 分离）的原因：单卡环境下 PD 分离通过 MinIO 间接传输 KV 数据，latency 高且不符合生产实际。Chunked prefill 模式同样具有 prefill 计算效率，且架构更简洁。

### 2.3 LMCache 部署模式

LMCache 以**嵌入库**形式运行在每个 vLLM 进程内（`LMCacheConnectorV1` kv connector），而非独立进程。这意味着：
- vLLM 崩溃时，**CPU cache 随进程消亡**
- GPU KV cache 本身也随进程消亡
- 只有 MinIO 中的持久化数据幸存

```
vLLM Process
├── API Server (FastAPI)
├── EngineCore (subprocess)
│   ├── LMCacheEngine (embedded)
│   │   ├── LocalCPUBackend   ← in-process dict, dies with process
│   │   ├── RemoteBackend     ← S3 connector (awscrt)
│   │   │   └── GMS MetadataClient (background TCP thread)
│   │   └── LMCacheConnectorV1
│   └── vLLM KV cache (GPU VRAM, PagedAttention)
└── ...
```

---

## 3. 核心组件详解

### 3.1 GMS（Global Metadata Service）

**文件**: `gms/gms_server.py`, `gms/metadata_client.py`

GMS 是自研的异步 TCP 元数据服务，实现了对每个 vLLM 进程 KV cache 状态的全局追踪。

#### 协议
```
Transport:  原始 TCP，持久连接
Encoding:   Line-delimited JSON
Heartbeat:  每 5 秒，客户端主动发送
Failure:    10 秒无心跳 → 标记 FAILED
Transfer:   FAILED 节点的序列转交给健康 peer
```

#### 核心数据结构

```
WorkerState:
  worker_id: str          # "dev:8000"
  host, port: str/int
  state: ALIVE | FAILED
  last_heartbeat: float

SequenceState:
  seq_id: str             # "dev:8000"（与 worker 同名）
  owner_worker_id: str
  epoch: int              # 单调递增，防止 stale 写入
  committed_frontier: int # 最后一个已提交 block 的索引
  block_order: List[int]  # chunk_hash 有序列表
  state: ACTIVE | PENDING_TRANSFER

BlockState:
  chunk_hash: int
  minio_node: str
  bucket: str
  object_key: str         # e.g. "Qwen/Qwen2.5-0.5B@1@0@3fa467f6304f3837@bfloat16"
  size_bytes: int
  state: PENDING | COMMITTED
```

#### 两阶段提交协议（2PC）

每个 KV block 写入 MinIO 前后，LMCache 需与 GMS 完成 2PC 握手：

```
Phase 1 — register_write:
  LMCache → GMS: { chunk_hash, seq_id, epoch, size_bytes }
  GMS 返回 ok=True 则继续 S3 PUT；
  返回 DEDUPLICATED（已有 COMMITTED 块）则跳过写入，节省带宽

Phase 2 — commit_write:
  LMCache → GMS: { chunk_hash, seq_id, epoch, minio_node, object_key, bucket }
  GMS 将 block 状态从 PENDING 升级为 COMMITTED
  推进 committed_frontier
```

#### 故障检测与 Failover

```
Monitor 循环 (每 2 秒):
  1. 扫描所有 Worker，idle > 10s → 标记 FAILED
  2. 扫描 ACTIVE 序列，owner 为 FAILED 时：
     a. 丢弃 PENDING 块（未完成写入，数据可能不完整）
     b. 保留所有 COMMITTED 块
     c. 将序列状态改为 PENDING_TRANSFER
     d. 将 owner 改为某个 ALIVE worker
     e. 下一次健康 worker 调用 get_pending_transfers() 时通知
```

### 3.2 RemoteBackend 的 GMS 集成

**文件**: `LMCache/lmcache/v1/storage_backend/remote_backend.py`

#### GMS 初始化（`_init_gms`）

```python
# 每个 vLLM 进程在 LMCache 初始化时连接 GMS
seq_id = f"{instance_id}:{gms_worker_port}"   # e.g. "dev:8000"
client = MetadataClient(worker_id=seq_id, ...)
client.start()                                 # 启动心跳线程

result = client.register_sequence(seq_id)
self._gms_epoch = result["epoch"]             # 获取当前 epoch

self._gms_recover_on_startup()                # 启动恢复线程（异步）
```

#### 热缓存恢复（`_gms_recover_worker`）

```python
def _gms_recover_worker(self):
    # 场景 1: 自身重启
    own = client.get_committed_blocks(self._gms_seq_id)
    # → GMS 返回本进程上一轮写入的所有 COMMITTED blocks

    # 场景 2: Peer failover（本进程接管崩溃节点的 cache）
    xfer = client.get_pending_transfers()
    # → GMS 返回已转移给本进程的序列列表

    for block in all_blocks:
        key = CacheEngineKey.from_string(block["object_key"])
        memory_obj = self.get_blocking(key)     # 从 MinIO 拉取（awscrt 异步 GET）
        local_cpu_backend.submit_put_task(key, memory_obj)   # 写入 CPU hot_cache
        memory_obj.ref_count_down()             # 释放调用方引用，hot_cache 为唯一持有者
```

#### KV Cache Key 设计

```
格式: {model_name}@{world_size}@{worker_id}@{chunk_hash_hex}@{dtype}
示例: Qwen/Qwen2.5-0.5B@1@0@3fa467f6304f3837@bfloat16

S3 存储时 / → _ 转义:
  MinIO key: Qwen_Qwen2.5-0.5B@1@0@3fa467f6304f3837@bfloat16

内容寻址: chunk_hash 由 token 序列 PYTHONHASHSEED=0 确定性计算
           → 相同 prompt 在不同进程重启后哈希不变
           → 不需要任何额外的映射表
```

### 3.3 内存引用计数

LMCache 对 CPU 内存使用引用计数进行 LRU 管理：

```
ref_count == 0: 内存可被立即回收
ref_count == 1: hot_cache 持有（唯一所有者），LRU 可驱逐
ref_count >= 2: 有调用方正在使用（pinned），LRU 不可驱逐

恢复流程中的 ref_count 流转:
  allocate()              → ref_count = 1
  submit_put_task()       → ref_count_up() → ref_count = 2   (hot_cache 持有)
  memory_obj.ref_count_down()              → ref_count = 1   (释放调用方引用)
  最终: hot_cache 为唯一持有者, ref_count = 1 ✓
```

---

## 4. RPC 通信架构

系统中存在三套独立的 RPC 机制：

| RPC 系统 | 传输层 | 编码 | 用途 |
|---------|--------|------|------|
| GMS 客户端 ↔ 服务端 | TCP 持久连接 | Line-delimited JSON | 元数据注册、心跳、故障通知 |
| LMCache Controller ↔ Worker | ZMQ TCP (Pull :8300 / Reply :8400) | msgpack | Cache 查询路由、跨实例 KV 传输 |
| LookupClient ↔ LookupServer | ZMQ IPC | msgpack | 进程内 cache 索引查询 |

---

## 5. 端到端测试

### 5.1 测试环境

```
硬件:  NVIDIA L4 GPU (23 GB VRAM)，单机
模型:  Qwen/Qwen2.5-0.5B（FP16 BFloat16）
Prompt: 72 tokens（固定文本，用于重现 cache hit）
Config: chunk_size=6, max_local_cpu_size=5GB, blocking_timeout=10s
```

### 5.2 测试流程

```
Step 1  启动服务
        MinIO :9000 | GMS :8500 | LMCache Controller :8100
        vLLM Instance 0 :8000  (GPU mem 40%)
        vLLM Instance 1 :8001  (GPU mem 40%)

Step 2  填充 cache（2 次推理请求 → 触发 MinIO 写入）
        写入: 11 blocks × 73,728 bytes = 808 KB

Step 3  Kill Instance 0
        kill -9 <main_pid> + kill -9 <EngineCore_child_pid>
        （注意: vLLM 用 multiprocessing 派生 EngineCore，
         仅 kill 主进程会留下孤儿子进程继续发送心跳）

Step 4  等待 GMS 故障检测（10s timeout + 2s monitor interval）
        dev:8000 状态变化:  ALIVE → FAILED
        序列转移:           dev:8000 owner → dev:8001（PENDING_TRANSFER）

Step 5  重启 Instance 0，观察 GMS recovery 日志

Step 6  比对有无预热时的 KV cache 获取耗时
```

### 5.3 测试结果

#### GMS 故障检测与 Failover（符合预期）

```json
// kill 后 ~12s，GMS 状态
Workers:
  dev:8000  state=FAILED   idle=55.6s
  dev:8001  state=ALIVE    idle=3.5s

Sequences:
  { seq_id: "dev:8000", owner: "dev:8001",
    epoch: 1, committed_frontier: 10,
    block_count: 11, state: "PENDING_TRANSFER" }
```

#### 重启后的 GMS Recovery 日志

```
02:07:28  LMCache Engine 初始化
02:07:32  GMS: connected  seq='dev:8000'  epoch=2  minio=localhost:9000
02:07:32  GMS recovery: self-restart  seq='dev:8000'  blocks=11
02:07:32  GMS recovery: preloading 11 blocks MinIO → CPU cache ...
02:07:32  GMS recovery: done — loaded=11  skipped=0          ← 仅 411ms
02:07:38  LMCache 初始化完成（CUDA graph capture 仍在进行）
02:08:07  首次推理请求到达
```

**恢复仅用 411ms，远早于 CUDA graph capture 完成（~16s）。**

#### KV Cache 获取耗时对比

| 场景 | 数据来源 | 获取耗时 | 备注 |
|------|---------|---------|------|
| 原始写入（首次推理） | GPU → MinIO PUT | 12.4 ms | 存储 66 tokens |
| **重启后首次请求（有预热）** | **CPU hot_cache** | **3.9 ms** | 66 tokens 全部命中 |
| 重启后第二次请求 | GPU KV cache | 0.5 ms | GPU 仍热 |
| 无预热时（估算，按需 MinIO GET） | MinIO GET × 11次 | ~19 ms | 11 × 1.7ms/block |

```
CPU cache hit  (有预热): 3.9 ms   ████
MinIO on-demand (无预热): ~19 ms  ███████████████████
                                  约 5× 性能差距
```

#### vLLM 层面的 cache hit rate

```
02:08:10  External prefix cache hit rate: 98.5%
          Inference Engine computed tokens: 0
          LMCache hit tokens: 66 / 66
```

重启后首批请求无需 GPU 重算 prefill，KV cache 完整恢复。

### 5.4 关键发现

**1. 孤儿进程问题**  
vLLM 以 `multiprocessing.Process` 启动 `EngineCore` 子进程（含 LMCache + GMS 心跳线程）。仅 kill 父进程（API Server）时，EngineCore 子进程作为孤儿继续运行，持续向 GMS 发送心跳，导致 GMS 无法检测到故障。必须同时 kill 父进程和子进程。

**2. 快速恢复掩盖了冷热差异**  
小模型（0.5B）+ 短 prompt 时，11 blocks 的 MinIO 预热仅需 411ms，而 CUDA graph capture 需 16s。因此 vLLM health endpoint 就绪时预热已完成，无法直接对比"有预热 vs 无预热"的 TTFT。在生产场景（大模型、长 prompt、上千 blocks）中，预热时间更长，重启后的前几批请求仍可能部分命中 MinIO，此时预热的价值更显著。

**3. Epoch 机制保证写入安全性**  
GMS 每次 failover 后递增 epoch（本次 0 → 1 → 2）。重启进程在 `register_sequence` 时获得最新 epoch，旧进程（若存在僵尸进程）的写入因 epoch 过期而被 GMS 拒绝，防止 stale 数据污染 MinIO。

---

## 6. 配置参考

### 6.1 LMCache 配置（lmcache_prefill.yaml）

```yaml
chunk_size: 6                          # 每个 cache 块覆盖的 token 数
local_cpu: True                        # 启用 CPU cache（必须为 True 才能预热）
max_local_cpu_size: 5.0                # CPU cache 上限 5 GB

remote_url: "s3://lmcache-bucket.localhost:9000"
remote_serde: "naive"                  # 不压缩（最低延迟）
blocking_timeout_secs: 10

enable_controller: True
lmcache_instance_id: "dev"
controller_pull_url: "localhost:8300"
controller_reply_url: "localhost:8400"
lmcache_worker_ports: [8050]

extra_config:
  s3_num_io_threads: 64
  disable_tls: True
  aws_access_key_id: "minioadmin"
  aws_secret_access_key: "minioadmin"
  gms_host: "localhost"
  gms_port: 8500
  gms_worker_host: "localhost"
  gms_worker_port: 8000              # 与 vllm --port 一致，用于 GMS seq_id
```

### 6.2 vLLM 启动参数

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-0.5B \
  --port 8000 \
  --gpu-memory-utilization 0.4 \
  --enable-chunked-prefill \
  --kv-transfer-config '{
    "kv_connector": "LMCacheConnectorV1",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
      "lmcache_config_file": ".../lmcache_prefill.yaml"
    }
  }'

export PYTHONPATH=/path/to/LMCache_minIO   # 使 gms/ 包可被 LMCache 导入
export PYTHONHASHSEED=0                     # 确保 chunk_hash 跨重启一致
```

---

## 7. 代码改动清单

### 7.1 核心修改：`remote_backend.py`

| 位置 | 改动 | 原因 |
|------|------|------|
| `_try_import_metadata_client()` | 移除硬编码 `sys.path.insert`，改用 PYTHONPATH 环境变量 | 消除路径硬编码，提高可移植性 |
| `_gms_recover_on_startup()` | 从空函数改为启动后台恢复线程 | 实现热缓存预热 |
| `_gms_recover_worker()`（新增） | 从 GMS 拉取 committed blocks，逐一从 MinIO 拉取写入 CPU cache | 核心预热逻辑 |
| `memory_obj.ref_count_down()` | 在 `submit_put_task` 后释放调用方引用 | 修复引用计数 bug，使 LRU 驱逐正常工作 |

### 7.2 配置修改

| 文件 | 改动 | 原因 |
|------|------|------|
| `lmcache_prefill.yaml` | `local_cpu: True`，`max_local_cpu_size: 5.0` | 启用 CPU cache（预热目标） |
| `lmcache_prefill.yaml` | `lmcache_worker_ports: [8050]` | 避免与 vLLM port 8001 冲突 |
| `lmcache_decode.yaml` | 同上（`[8051]`）| 同上 |
| `start_prefill.sh` / `start_decode.sh` | `--enable-chunked-prefill`，`export PYTHONPATH` | 对等架构 + GMS 包可导入 |

---

## 8. 局限性与后续优化方向

| 问题 | 现状 | 优化方向 |
|------|------|---------|
| CPU cache 随进程消亡 | CPU cache 在 vLLM 进程内，kill 后全部丢失 | 使用 LMCache MPCacheEngine 模式（独立进程），CPU cache 跨 vLLM 重启存活 |
| 预热与服务同时竞争 MinIO I/O | 恢复线程和推理请求同时从 MinIO 拉取数据 | 限速（rate limit）或优先级队列；先预热后开放流量 |
| 单机无 RDMA | awscrt over TCP 到本地 MinIO，带宽受限 | 多机部署时使用 RDMA/NIXL（需 InfiniBand 或 RoCE 网卡） |
| 孤儿进程需手动 kill | vLLM EngineCore 子进程独立存活 | 使用 `process group` kill，或监控父进程 PID 使子进程自动退出 |
| Peer failover 预热未触发 | 当前测试中 dev:8001 未消费转移来的 dev:8000 序列 | 触发条件：dev:8001 的 `get_pending_transfers()` 在其 `_gms_recover_worker` 中调用，需 dev:8001 重启才会消费 |
