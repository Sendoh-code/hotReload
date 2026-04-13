# LMCache + MinIO Hot-Cache Fault-Tolerant Recovery System
## Technical Report

**Date**: 2026-04-13  
**Environment**: Single-node NVIDIA L4 (23 GB VRAM)  
**Model**: Qwen/Qwen2.5-0.5B  

---

## 1. Background and Objectives

### 1.1 The Problem

In multi-instance vLLM deployments, each process maintains KV cache (key-value cache) in memory. When a vLLM process crashes and restarts, its entire in-memory KV cache is lost. The first batch of requests after a restart must recompute prefill from scratch, causing a significant TTFT (Time To First Token) regression.

### 1.2 Design Goals

- **Fault tolerance**: KV cache is persisted in object storage (MinIO); a process crash does not cause permanent data loss.
- **Warm restart**: Upon restart, proactively preload KV cache from MinIO into local CPU memory so that the first requests hit the local cache rather than the remote object store.
- **Non-intrusiveness**: All of this logic lives entirely within the LMCache layer — no invasive modifications to vLLM's core code.
- **Observability**: The GMS (Global Metadata Service) tracks the set of cache blocks owned by each process cluster-wide.

---

## 2. System Architecture

### 2.1 Overall Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Single Node (NVIDIA L4)                          │
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
│                         │ TCP + Line-delimited JSON                 │
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
│  │  Each block: 73,728 bytes  (chunk_size = 6 tokens)        │     │
│  └───────────────────────────────────────────────────────────┘     │
│                                                                     │
│  ┌───────────────────────┐                                         │
│  │  LMCache Controller   │                                         │
│  │  HTTP  :8100          │                                         │
│  │  ZMQ Pull  :8300      │                                         │
│  │  ZMQ Reply :8400      │                                         │
│  └───────────────────────┘                                         │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Symmetric Instance Design

Unlike traditional PD separation (dedicated Prefill and Decode roles), both instances in this project are **fully symmetric peers**. Each runs with `--enable-chunked-prefill` and can independently handle a complete request (prefill + decode).

| Property | Instance 0 | Instance 1 |
|----------|------------|------------|
| vLLM HTTP port | 8000 | 8001 |
| GPU memory utilization | 40% | 40% |
| LMCache config | lmcache_prefill.yaml | lmcache_decode.yaml |
| LMCache Worker ZMQ | :8050 | :8051 |
| GMS sequence ID | `dev:8000` | `dev:8001` |
| Role | Symmetric peer | Symmetric peer |

> **Why symmetric rather than PD-separated?** On a single GPU, PD separation transfers KV data indirectly through MinIO, introducing high latency that is impractical for production. Chunked prefill achieves equivalent computation efficiency without the architectural complexity.

### 2.3 LMCache Deployment Model

LMCache runs as an **embedded library** inside each vLLM process (via the `LMCacheConnectorV1` kv connector), not as a standalone service. This has an important implication:

- When vLLM crashes, the **CPU cache dies with the process**
- Only data persisted in MinIO survives a crash

```
vLLM Process
├── API Server (FastAPI)
└── EngineCore (subprocess via multiprocessing)
    ├── LMCacheEngine (embedded library)
    │   ├── LocalCPUBackend   ← in-process Python dict; lost on crash
    │   ├── RemoteBackend     ← S3 connector (awscrt)
    │   │   └── GMS MetadataClient (background TCP heartbeat thread)
    │   └── LMCacheConnectorV1
    └── vLLM KV cache (GPU VRAM, PagedAttention)
```

---

## 3. Core Components

### 3.1 GMS — Global Metadata Service

**Files**: `gms/gms_server.py`, `gms/metadata_client.py`

GMS is a custom asyncio TCP metadata service that maintains a global view of each vLLM process's KV cache state.

#### Wire Protocol

```
Transport:  Raw TCP, persistent connection
Encoding:   Line-delimited JSON  (one JSON object per line)
Heartbeat:  Client sends heartbeat every 5 seconds
Failure:    Worker declared FAILED after 10 seconds of missed heartbeats
Transfer:   Failed worker's sequences are reassigned to a healthy peer
```

#### Core Data Structures

```
WorkerState
  worker_id:       str          # e.g. "dev:8000"
  host, port:      str / int
  state:           ALIVE | FAILED
  last_heartbeat:  float (Unix timestamp)

SequenceState
  seq_id:               str     # matches worker_id by convention
  owner_worker_id:      str
  epoch:                int     # monotonically increasing; prevents stale writes
  committed_frontier:   int     # index of the last COMMITTED block
  block_order:          List[int]   # ordered chunk_hash list
  state:                ACTIVE | PENDING_TRANSFER

BlockState
  chunk_hash:    int
  minio_node:    str            # e.g. "localhost:9000"
  bucket:        str
  object_key:    str            # e.g. "Qwen/Qwen2.5-0.5B@1@0@3fa467f6304f3837@bfloat16"
  size_bytes:    int
  state:         PENDING | COMMITTED
```

#### Two-Phase Commit Protocol (2PC)

Before and after each KV block is written to MinIO, LMCache performs a 2PC handshake with GMS:

```
Phase 1 — register_write
  LMCache → GMS: { chunk_hash, seq_id, epoch, size_bytes }
  GMS returns ok=True   → proceed with the S3 PUT
  GMS returns DEDUPLICATED (block already COMMITTED) → skip the PUT, saving bandwidth

Phase 2 — commit_write
  LMCache → GMS: { chunk_hash, seq_id, epoch, minio_node, object_key, bucket }
  GMS promotes the block from PENDING to COMMITTED
  GMS advances committed_frontier for the sequence
```

#### Failure Detection and Failover

```
Monitor loop (every 2 seconds):
  1. Scan all workers: idle > 10 s → mark FAILED
  2. For each ACTIVE sequence whose owner is now FAILED:
       a. Discard PENDING blocks (write was incomplete; data may be corrupt)
       b. Retain all COMMITTED blocks
       c. Set sequence state to PENDING_TRANSFER
       d. Reassign ownership to a healthy peer worker
       e. Notify the new owner on next call to get_pending_transfers()
```

### 3.2 GMS Integration in RemoteBackend

**File**: `LMCache/lmcache/v1/storage_backend/remote_backend.py`

#### Initialization (`_init_gms`)

```python
# Each vLLM process connects to GMS during LMCache initialization
seq_id = f"{instance_id}:{gms_worker_port}"   # e.g. "dev:8000"
client = MetadataClient(worker_id=seq_id, ...)
client.start()                                 # starts the background heartbeat thread

result = client.register_sequence(seq_id)
self._gms_epoch = result["epoch"]             # receive current epoch

self._gms_recover_on_startup()                # launch async recovery thread
```

#### Hot Cache Recovery (`_gms_recover_worker`)

```python
def _gms_recover_worker(self):
    # Scenario 1: self-restart — reclaim this process's own previous blocks
    own = client.get_committed_blocks(self._gms_seq_id)
    # GMS returns all COMMITTED blocks from this process's last run

    # Scenario 2: peer failover — claim blocks transferred from a crashed peer
    xfer = client.get_pending_transfers()
    # GMS returns sequences newly assigned to this worker by failover

    for block in all_blocks:
        key = CacheEngineKey.from_string(block["object_key"])
        memory_obj = self.get_blocking(key)          # fetch from MinIO (awscrt async GET)
        local_cpu_backend.submit_put_task(key, memory_obj)  # register in CPU hot_cache
        memory_obj.ref_count_down()   # release caller's ref; hot_cache becomes sole owner
```

#### KV Cache Key Design

```
Format:  {model_name}@{world_size}@{worker_id}@{chunk_hash_hex}@{dtype}
Example: Qwen/Qwen2.5-0.5B@1@0@3fa467f6304f3837@bfloat16

MinIO storage: "/" in model name is replaced with "_" by the S3 connector's
               _format_safe_path() helper:
  MinIO key:   Qwen_Qwen2.5-0.5B@1@0@3fa467f6304f3837@bfloat16

Content-addressing: chunk_hash is deterministically computed from the token
                    sequence (PYTHONHASHSEED=0 makes it stable across restarts).
                    No extra mapping table is needed.
```

### 3.3 Memory Reference Counting

LMCache uses reference counting to drive LRU eviction of CPU cache entries:

```
ref_count == 0: memory may be reclaimed immediately
ref_count == 1: held by hot_cache only (sole owner); LRU-evictable
ref_count >= 2: an active caller holds a reference (pinned); not evictable

Reference count flow during recovery:
  allocate()                → ref_count = 1   (allocated for download buffer)
  submit_put_task()         → ref_count_up()  → ref_count = 2  (hot_cache acquires)
  memory_obj.ref_count_down()                 → ref_count = 1  (caller releases)
  Final state: hot_cache is the sole owner, ref_count = 1  ✓

Without the ref_count_down() call, ref_count would stay at 2 permanently,
making the block invisible to the LRU eviction policy — a memory leak.
```

---

## 4. RPC Communication Architecture

Three independent RPC mechanisms coexist in the system:

| RPC System | Transport | Encoding | Purpose |
|-----------|-----------|----------|---------|
| GMS Client ↔ Server | Raw TCP, persistent | Line-delimited JSON | Metadata registration, heartbeat, failover notification |
| LMCache Controller ↔ Worker | ZMQ TCP (Pull :8300 / Reply :8400) | msgpack | Cache lookup routing, cross-instance KV transfer |
| LookupClient ↔ LookupServer | ZMQ IPC | msgpack | In-process cache index queries |

---

## 5. End-to-End Validation

### 5.1 Test Environment

```
Hardware:   Single NVIDIA L4 GPU (23 GB VRAM)
Model:      Qwen/Qwen2.5-0.5B  (BFloat16)
Prompt:     72 tokens (fixed text to ensure reproducible cache hits)
Config:     chunk_size=6, max_local_cpu_size=5 GB, blocking_timeout=10 s
```

### 5.2 Test Procedure

```
Step 1  Start all services
        MinIO :9000 | GMS :8500 | LMCache Controller :8100
        vLLM Instance 0 :8000  (40% GPU memory)
        vLLM Instance 1 :8001  (40% GPU memory)

Step 2  Populate cache (2 inference requests → trigger MinIO write)
        Written: 11 blocks × 73,728 bytes = 808 KB

Step 3  Kill Instance 0
        kill -9 <main_pid> + kill -9 <EngineCore_child_pid>
        Note: vLLM forks EngineCore as a child process; killing only the
        parent leaves the child alive as an orphan, still sending heartbeats.
        Both must be killed.

Step 4  Wait for GMS failure detection  (10 s timeout + 2 s monitor interval)
        dev:8000 transitions:  ALIVE → FAILED
        Sequence transfer:     dev:8000 reassigned to dev:8001  (PENDING_TRANSFER)

Step 5  Restart Instance 0; observe GMS recovery log

Step 6  Compare KV cache retrieval latency: with vs. without CPU preloading
```

### 5.3 Test Results

#### GMS Failure Detection and Failover (as expected)

```json
// GMS state ~12 s after kill
Workers:
  dev:8000  state=FAILED   idle=55.6 s
  dev:8001  state=ALIVE    idle=3.5 s

Sequences:
  { "seq_id": "dev:8000", "owner_worker_id": "dev:8001",
    "epoch": 1, "committed_frontier": 10,
    "block_count": 11, "state": "PENDING_TRANSFER" }
```

#### GMS Recovery Log After Restart

```
02:07:28  LMCacheEngine initialization begins
02:07:32  GMS: connected  seq='dev:8000'  epoch=2  minio=localhost:9000
02:07:32  GMS recovery: self-restart  seq='dev:8000'  blocks=11
02:07:32  GMS recovery: preloading 11 blocks MinIO → CPU cache ...
02:07:32  GMS recovery: done — loaded=11  skipped=0     ← completed in 411 ms
02:07:38  LMCache fully initialized  (CUDA graph capture still in progress)
02:08:07  First inference request served
```

**The entire recovery completed in 411 ms, well before CUDA graph capture finished (~16 s).**

#### KV Cache Retrieval Latency Comparison

| Scenario | Data Source | Latency | Notes |
|----------|------------|---------|-------|
| Original write (first inference) | GPU → MinIO PUT | 12.4 ms | Store 66 tokens |
| **Post-restart, 1st request (with preloading)** | **CPU hot_cache** | **3.9 ms** | 66/66 tokens hit |
| Post-restart, 2nd request | GPU KV cache | 0.5 ms | GPU still hot |
| Without preloading (estimated, on-demand MinIO GET) | MinIO GET × 11 | ~19 ms | 11 × ~1.7 ms/block |

```
CPU cache hit  (with preloading):    3.9 ms  ████
MinIO on-demand (without preloading): ~19 ms  ███████████████████
                                              ~5× latency reduction
```

#### vLLM-Level Cache Hit Rate

```
02:08:10  External prefix cache hit rate: 98.5%
          Inference Engine computed tokens: 0
          LMCache hit tokens: 66 / 66
```

The first batch of requests after restart required zero GPU prefill recomputation. KV cache was fully restored.

### 5.4 Key Findings

**1. Orphan Process Issue**  
vLLM spawns `EngineCore` as a child process via `multiprocessing.Process`. The LMCache engine (including the GMS heartbeat thread) runs inside EngineCore. When only the parent API Server process is killed, EngineCore survives as an orphan and continues sending heartbeats to GMS, preventing failure detection. **Both the parent and the EngineCore child process must be killed** to trigger GMS failover.

**2. Fast Recovery Masks the Cold/Warm Contrast**  
With a small model (0.5B parameters) and a short prompt, preloading 11 blocks takes only 411 ms, far less than the CUDA graph capture duration (~16 s). By the time the vLLM health endpoint becomes available, preloading is already complete — making it impossible to measure a cold first-request in this configuration. In production scenarios with larger models, longer prompts, and hundreds or thousands of cached blocks, preloading takes proportionally longer and the warm-up benefit is more pronounced.

**3. Epoch Fencing Ensures Write Safety**  
GMS increments the epoch on every failover (in this test: 0 → 1 → 2). The restarted process receives the latest epoch during `register_sequence`. Any stale writes from a zombie process are rejected by GMS due to epoch mismatch, preventing corrupt or outdated data from reaching MinIO.

---

## 6. Configuration Reference

### 6.1 LMCache Configuration (`lmcache_prefill.yaml`)

```yaml
chunk_size: 6                          # tokens per cache block
local_cpu: True                        # must be True to enable CPU preloading
max_local_cpu_size: 5.0                # CPU cache capacity limit (GB)

remote_url: "s3://lmcache-bucket.localhost:9000"
remote_serde: "naive"                  # no compression (lowest latency)
blocking_timeout_secs: 10

enable_controller: True
lmcache_instance_id: "dev"
controller_pull_url: "localhost:8300"
controller_reply_url: "localhost:8400"
lmcache_worker_ports: [8050]           # must not conflict with vLLM HTTP ports

extra_config:
  s3_num_io_threads: 64
  disable_tls: True
  aws_access_key_id: "minioadmin"
  aws_secret_access_key: "minioadmin"
  gms_host: "localhost"
  gms_port: 8500
  gms_worker_host: "localhost"
  gms_worker_port: 8000                # must match vLLM --port; used as GMS seq_id
```

### 6.2 vLLM Launch Arguments

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

# Required environment variables
export PYTHONPATH=/path/to/LMCache_minIO   # makes the gms/ package importable by LMCache
export PYTHONHASHSEED=0                     # ensures chunk_hash is stable across restarts
```

---

## 7. Code Change Summary

### 7.1 Core Modification: `remote_backend.py`

| Location | Change | Reason |
|----------|--------|--------|
| `_try_import_metadata_client()` | Removed hardcoded `sys.path.insert`; rely on `PYTHONPATH` instead | Eliminate path hardcoding; improve portability |
| `_gms_recover_on_startup()` | Changed from a no-op to launching a background recovery thread | Implement hot cache preloading |
| `_gms_recover_worker()` (new) | Pulls committed block list from GMS; fetches each from MinIO; writes into CPU hot_cache | Core preloading logic |
| `memory_obj.ref_count_down()` | Added after `submit_put_task()` to release the caller's reference | Fix reference count bug; allow LRU eviction to work correctly |

### 7.2 Configuration Changes

| File | Change | Reason |
|------|--------|--------|
| `lmcache_prefill.yaml` | `local_cpu: True`, `max_local_cpu_size: 5.0` | Enable CPU cache (preloading target) |
| `lmcache_prefill.yaml` | `lmcache_worker_ports: [8050]` | Avoid conflict with vLLM port 8001 |
| `lmcache_decode.yaml` | Same (`[8051]`) | Same reason |
| `start_prefill.sh` / `start_decode.sh` | Added `--enable-chunked-prefill`, `export PYTHONPATH` | Symmetric architecture + GMS package importable |

---

## 8. Limitations and Future Work

| Issue | Current State | Proposed Improvement |
|-------|--------------|----------------------|
| CPU cache lost on process crash | CPU cache lives inside the vLLM process; crash erases it | Use LMCache's `MPCacheEngine` mode (separate process); CPU cache survives vLLM restarts |
| Recovery I/O competes with inference | Recovery thread and inference requests both pull from MinIO simultaneously | Rate-limit recovery downloads or use a priority queue; optionally delay traffic until preloading is complete |
| No RDMA on single-node L4 | awscrt over TCP to local MinIO; bandwidth is CPU-bound | Multi-node deployment with InfiniBand or RoCE + NIXL/RDMA for direct GPU-to-GPU KV transfer |
| Orphan processes require manual kill | vLLM EngineCore child survives parent kill | Kill the entire process group; or have the child monitor its parent PID and self-exit |
| Peer failover preloading not triggered | `get_pending_transfers()` in `_gms_recover_worker` is only called on restart; a live peer never consumes transferred sequences | Implement a periodic poll in the background thread so a running peer can absorb transferred blocks without restarting |
