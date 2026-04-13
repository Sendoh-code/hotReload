"""
gms_server.py — Global Metadata Service (GMS) Server
=====================================================
GFS-inspired control-plane for distributed KV cache management.

Responsibilities:
  - Block Location Index  : chunk_hash → (minio_node, bucket, object_key, state)
  - Ownership Registry    : seq_id → (owner_worker_id, epoch, committed_frontier)
  - Worker Registry       : heartbeat-based liveness tracking
  - 2PC Write Protocol    : PENDING → COMMITTED state machine
  - Epoch Fencing         : rejects stale writes from evicted workers
  - Failure Detection     : heartbeat timeout → ownership transfer

Protocol: line-delimited JSON over TCP (one JSON object per line)
  Request:  {"id": "<uuid>", "method": "<name>", "params": {...}}
  Response: {"id": "<uuid>", "ok": true|false, "error": "<msg>", ...result fields}

Port: 8500 (avoids conflicts with LMCache 8100/8200/8300/8400, MinIO 9000/9001)

RPCs:
  register_worker        worker_id, host, port
  heartbeat              worker_id
  register_sequence      seq_id, worker_id
  lookup                 chunk_hashes: [int]
  register_write         chunk_hash, seq_id, worker_id, epoch, size_bytes   [2PC Phase 1]
  commit_write           chunk_hash, seq_id, worker_id, epoch,              [2PC Phase 2]
                           minio_node, object_key, bucket, size_bytes
  report_access          worker_id, chunk_hashes: [int]
  get_committed_blocks   seq_id                                              [failover]
  get_pending_transfers  worker_id                                           [failover]
  list_workers           —                                                   [debug]
  list_sequences         —                                                   [debug]
  list_blocks            limit                                               [debug]
"""

import asyncio
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [GMS] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("gms")

# ─── Tunables ────────────────────────────────────────────────────────────────

GMS_HOST = "0.0.0.0"
GMS_PORT = 8500

HEARTBEAT_INTERVAL = 5    # seconds; workers must send heartbeat this often
FAILURE_TIMEOUT    = 10   # seconds; GMS declares failure after this idle time
MONITOR_INTERVAL   = 2    # seconds; how often GMS polls for failed workers


# ─── Data Structures ─────────────────────────────────────────────────────────

@dataclass
class BlockRecord:
    """One KV cache block tracked by GMS. Content-addressed by chunk_hash."""
    chunk_hash: int

    # Location (filled in at commit_write; empty string while PENDING)
    minio_node: str
    bucket: str
    object_key: str
    size_bytes: int

    # Ownership
    owner_worker_id: str
    seq_id: str
    epoch: int                  # epoch at the time this block was registered

    # State machine: PENDING → COMMITTED | DISCARDED
    state: str                  # "PENDING" | "COMMITTED" | "DISCARDED"

    # Access tracking (for future admission gate integration)
    created_at: float
    last_accessed: float
    access_count: int = 1
    refcount: int = 1           # >1 when multiple workers share a deduplicated block

    def to_dict(self) -> dict:
        return {
            "chunk_hash":       self.chunk_hash,
            "minio_node":       self.minio_node,
            "bucket":           self.bucket,
            "object_key":       self.object_key,
            "size_bytes":       self.size_bytes,
            "owner_worker_id":  self.owner_worker_id,
            "seq_id":           self.seq_id,
            "epoch":            self.epoch,
            "state":            self.state,
            "refcount":         self.refcount,
        }


@dataclass
class SequenceRecord:
    """
    Ownership record for one decode sequence (one ongoing LLM request).
    Per-sequence ownership: only one worker writes at any time.
    """
    seq_id: str
    owner_worker_id: str
    epoch: int                  # monotonically increasing; incremented on each ownership transfer

    # Ordered list of chunk_hashes produced by this sequence (append-only)
    block_order: List[int] = field(default_factory=list)

    # Index into block_order of the last COMMITTED block (-1 = none committed yet)
    committed_frontier: int = -1

    # "ACTIVE" | "PENDING_TRANSFER" | "RECOVERED" | "FAILED"
    state: str = "ACTIVE"

    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "seq_id":               self.seq_id,
            "owner_worker_id":      self.owner_worker_id,
            "epoch":                self.epoch,
            "committed_frontier":   self.committed_frontier,
            "block_count":          len(self.block_order),
            "state":                self.state,
        }


@dataclass
class WorkerRecord:
    """Liveness record for one vLLM worker instance."""
    worker_id: str
    host: str
    port: int
    last_heartbeat: float
    state: str = "ALIVE"        # "ALIVE" | "FAILED"
    registered_at: float = field(default_factory=time.time)


# ─── In-Memory GMS State ─────────────────────────────────────────────────────

class GMSState:
    def __init__(self):
        # Primary indexes
        self.blocks:    Dict[int, BlockRecord]      = {}   # chunk_hash → BlockRecord
        self.sequences: Dict[str, SequenceRecord]   = {}   # seq_id     → SequenceRecord
        self.workers:   Dict[str, WorkerRecord]     = {}   # worker_id  → WorkerRecord

        # Reverse index: worker_id → set of seq_ids it currently owns
        self.worker_sequences: Dict[str, Set[str]] = defaultdict(set)


# ─── Business Logic ──────────────────────────────────────────────────────────

class GMSHandlers:
    """
    Pure business logic: each method receives a params dict and returns a result dict.
    All mutations go through this class; the server layer handles only I/O.
    """

    def __init__(self, state: GMSState):
        self._s = state

    def dispatch(self, method: str, params: dict) -> dict:
        table = {
            "register_worker":       self._register_worker,
            "heartbeat":             self._heartbeat,
            "register_sequence":     self._register_sequence,
            "lookup":                self._lookup,
            "register_write":        self._register_write,
            "commit_write":          self._commit_write,
            "report_access":         self._report_access,
            "get_committed_blocks":  self._get_committed_blocks,
            "get_pending_transfers": self._get_pending_transfers,
            "list_workers":          self._list_workers,
            "list_sequences":        self._list_sequences,
            "list_blocks":           self._list_blocks,
        }
        fn = table.get(method)
        if fn is None:
            return {"ok": False, "error": f"Unknown method: {method!r}"}
        try:
            return fn(params)
        except KeyError as e:
            return {"ok": False, "error": f"Missing required param: {e}"}
        except Exception as e:
            logger.exception(f"Handler {method!r} raised exception")
            return {"ok": False, "error": str(e)}

    # ── Worker Registration & Heartbeat ──────────────────────────────────────

    def _register_worker(self, p: dict) -> dict:
        worker_id = p["worker_id"]
        host      = p.get("host", "unknown")
        port      = int(p.get("port", 0))
        now       = time.time()

        if worker_id in self._s.workers:
            # Worker restart: update fields, revive state
            w = self._s.workers[worker_id]
            w.host            = host
            w.port            = port
            w.last_heartbeat  = now
            w.state           = "ALIVE"
            logger.info(f"Worker re-registered: {worker_id} at {host}:{port}")
        else:
            self._s.workers[worker_id] = WorkerRecord(
                worker_id=worker_id, host=host, port=port, last_heartbeat=now
            )
            logger.info(f"Worker registered: {worker_id} at {host}:{port}")

        return {"ok": True, "worker_id": worker_id, "timestamp": now}

    def _heartbeat(self, p: dict) -> dict:
        worker_id = p["worker_id"]
        w = self._s.workers.get(worker_id)
        if w is None:
            # Auto-register on first heartbeat (handles race at startup)
            return self._register_worker(p)

        now = time.time()
        w.last_heartbeat = now
        if w.state == "FAILED":
            w.state = "ALIVE"
            logger.info(f"Worker revived via heartbeat: {worker_id}")
        return {"ok": True, "timestamp": now}

    # ── Sequence Ownership ───────────────────────────────────────────────────

    def _register_sequence(self, p: dict) -> dict:
        seq_id    = p["seq_id"]
        worker_id = p["worker_id"]

        # Worker must be registered and alive
        w = self._s.workers.get(worker_id)
        if w is None:
            return {"ok": False, "error": f"Worker not registered: {worker_id}"}
        if w.state == "FAILED":
            return {"ok": False, "error": f"Worker is FAILED: {worker_id}"}

        if seq_id in self._s.sequences:
            seq = self._s.sequences[seq_id]
            if seq.owner_worker_id == worker_id:
                # Idempotent: same worker re-registering
                return {"ok": True, "epoch": seq.epoch, "seq_id": seq_id}

            # Allow the original worker to reclaim its sequence after restart,
            # even if GMS already transferred it to another worker.
            # seq_id = "instance_id:port", so only the same process (same port)
            # can reclaim.  We bump the epoch to fence out any in-flight writes
            # from the temporary owner.
            if seq.state in ("PENDING_TRANSFER", "RECOVERED"):
                old_owner = seq.owner_worker_id
                self._s.worker_sequences.get(old_owner, set()).discard(seq_id)
                seq.owner_worker_id = worker_id
                seq.state           = "ACTIVE"
                seq.epoch          += 1
                self._s.worker_sequences[worker_id].add(seq_id)
                logger.info(
                    f"Sequence reclaimed: {seq_id!r}  {old_owner} → {worker_id}  "
                    f"new epoch={seq.epoch}"
                )
                return {"ok": True, "epoch": seq.epoch, "seq_id": seq_id}

            return {
                "ok":    False,
                "error": f"Sequence {seq_id!r} already owned by {seq.owner_worker_id!r}",
            }

        seq = SequenceRecord(seq_id=seq_id, owner_worker_id=worker_id, epoch=0)
        self._s.sequences[seq_id] = seq
        self._s.worker_sequences[worker_id].add(seq_id)
        logger.info(f"Sequence registered: {seq_id!r} → {worker_id}")
        return {"ok": True, "epoch": 0, "seq_id": seq_id}

    # ── Block Lookup ─────────────────────────────────────────────────────────

    def _lookup(self, p: dict) -> dict:
        chunk_hashes = p["chunk_hashes"]
        now  = time.time()
        hits, misses = [], []

        for ch in chunk_hashes:
            block = self._s.blocks.get(ch)
            if block is not None and block.state == "COMMITTED":
                block.last_accessed = now
                block.access_count  += 1
                hits.append(block.to_dict())
            else:
                misses.append(ch)

        return {"ok": True, "hits": hits, "misses": misses}

    # ── 2PC Write Protocol ───────────────────────────────────────────────────

    def _register_write(self, p: dict) -> dict:
        """
        Phase 1 of 2PC: worker declares intent to write a block.

        Enforces epoch fencing: returns ok=False if worker_epoch < seq.epoch,
        meaning this worker has already been evicted and must stop writing.

        Handles deduplication: if the block already exists (COMMITTED), increments
        refcount and returns deduplicated=True so the worker can skip the S3 PUT.
        """
        chunk_hash   = int(p["chunk_hash"])
        seq_id       = p["seq_id"]
        worker_id    = p["worker_id"]
        worker_epoch = int(p["epoch"])
        size_bytes   = int(p.get("size_bytes", 0))

        # Auto-create sequence if this is the first write for a new request
        if seq_id not in self._s.sequences:
            result = self._register_sequence({"seq_id": seq_id, "worker_id": worker_id})
            if not result["ok"]:
                return result

        seq = self._s.sequences[seq_id]

        # ── Epoch fencing ────────────────────────────────────────────────────
        if worker_epoch < seq.epoch:
            return {
                "ok":            False,
                "reason":        "STALE_EPOCH",
                "error":         (
                    f"Stale epoch for seq {seq_id!r}: "
                    f"worker={worker_epoch}, current={seq.epoch}"
                ),
                "current_epoch": seq.epoch,
            }

        # ── Deduplication check ──────────────────────────────────────────────
        existing = self._s.blocks.get(chunk_hash)
        if existing is not None and existing.state == "COMMITTED":
            existing.refcount     += 1
            existing.last_accessed = time.time()
            return {
                "ok":           True,
                "deduplicated": True,
                "reason":       "ALREADY_COMMITTED",
            }

        # ── Create PENDING record ────────────────────────────────────────────
        now = time.time()
        self._s.blocks[chunk_hash] = BlockRecord(
            chunk_hash=chunk_hash,
            minio_node="",          # will be filled in at commit_write
            bucket="",
            object_key="",
            size_bytes=size_bytes,
            owner_worker_id=worker_id,
            seq_id=seq_id,
            epoch=worker_epoch,
            state="PENDING",
            created_at=now,
            last_accessed=now,
        )

        # Track block order in the sequence (append-only, no duplicates)
        if chunk_hash not in seq.block_order:
            seq.block_order.append(chunk_hash)
        seq.updated_at = now

        return {"ok": True, "deduplicated": False}

    def _commit_write(self, p: dict) -> dict:
        """
        Phase 2 of 2PC: worker confirms the S3 PUT succeeded.

        Promotes block state PENDING → COMMITTED, records the actual MinIO
        location, and advances the committed_frontier for the sequence.
        """
        chunk_hash   = int(p["chunk_hash"])
        seq_id       = p["seq_id"]
        worker_id    = p["worker_id"]
        worker_epoch = int(p["epoch"])
        minio_node   = p["minio_node"]
        object_key   = p["object_key"]
        bucket       = p.get("bucket", "kvcache")
        size_bytes   = int(p.get("size_bytes", 0))

        seq = self._s.sequences.get(seq_id)
        if seq is None:
            return {"ok": False, "error": f"Unknown sequence: {seq_id!r}"}

        # ── Epoch fencing ────────────────────────────────────────────────────
        if worker_epoch < seq.epoch:
            return {
                "ok":            False,
                "reason":        "STALE_EPOCH",
                "error":         (
                    f"Stale epoch for seq {seq_id!r}: "
                    f"worker={worker_epoch}, current={seq.epoch}"
                ),
                "current_epoch": seq.epoch,
            }

        block = self._s.blocks.get(chunk_hash)
        if block is None:
            return {
                "ok":    False,
                "error": f"Block {chunk_hash} not registered (call register_write first)",
            }
        if block.state == "COMMITTED":
            # Idempotent: already committed (duplicate commit_write call)
            return {"ok": True, "committed_frontier": seq.committed_frontier, "deduplicated": True}

        # ── Promote to COMMITTED ─────────────────────────────────────────────
        block.minio_node   = minio_node
        block.bucket       = bucket
        block.object_key   = object_key
        block.size_bytes   = size_bytes
        block.state        = "COMMITTED"
        block.last_accessed = time.time()

        # Advance committed_frontier to the index of this block in the sequence
        try:
            idx = seq.block_order.index(chunk_hash)
            if idx > seq.committed_frontier:
                seq.committed_frontier = idx
        except ValueError:
            pass  # block not in this sequence's order (shouldn't happen)

        seq.updated_at = time.time()
        logger.debug(
            f"Block committed: seq={seq_id!r} "
            f"hash={chunk_hash:016x} frontier={seq.committed_frontier}"
        )
        return {"ok": True, "committed_frontier": seq.committed_frontier}

    # ── Access Reporting ─────────────────────────────────────────────────────

    def _report_access(self, p: dict) -> dict:
        chunk_hashes = p.get("chunk_hashes", [])
        now = time.time()
        updated = 0
        for ch in chunk_hashes:
            block = self._s.blocks.get(int(ch))
            if block is not None:
                block.last_accessed = now
                block.access_count  += 1
                updated             += 1
        return {"ok": True, "updated": updated}

    # ── Failover Recovery ────────────────────────────────────────────────────

    def _get_committed_blocks(self, p: dict) -> dict:
        """
        Return all COMMITTED blocks for a sequence, ordered and bounded by
        committed_frontier. Called by the new owner on failover to reload
        the hot cache from MinIO.
        """
        seq_id = p["seq_id"]
        seq = self._s.sequences.get(seq_id)
        if seq is None:
            return {"ok": False, "error": f"Unknown sequence: {seq_id!r}"}

        blocks = []
        for idx, ch in enumerate(seq.block_order):
            if idx > seq.committed_frontier:
                break                               # stop at frontier
            block = self._s.blocks.get(ch)
            if block is not None and block.state == "COMMITTED":
                blocks.append({
                    "index":        idx,
                    "chunk_hash":   ch,
                    "minio_node":   block.minio_node,
                    "bucket":       block.bucket,
                    "object_key":   block.object_key,
                    "size_bytes":   block.size_bytes,
                })

        return {
            "ok":                 True,
            "seq_id":             seq_id,
            "owner_worker_id":    seq.owner_worker_id,
            "epoch":              seq.epoch,
            "committed_frontier": seq.committed_frontier,
            "blocks":             blocks,
        }

    def _get_pending_transfers(self, p: dict) -> dict:
        """
        Worker polls this after startup or after a suspected failover to check
        whether it has been assigned any sequences from a failed peer.
        Each sequence is returned once; subsequent calls will not include it
        again (state transitions PENDING_TRANSFER → RECOVERED).
        """
        worker_id = p["worker_id"]
        transferred = []

        for seq_id, seq in self._s.sequences.items():
            if seq.state != "PENDING_TRANSFER":
                continue

            if seq.owner_worker_id == worker_id:
                # 已明确分配给本 worker
                seq.state = "RECOVERED"
                transferred.append({
                    "seq_id":             seq_id,
                    "epoch":              seq.epoch,
                    "committed_frontier": seq.committed_frontier,
                })

            elif seq.owner_worker_id == "":
                # 孤儿序列（崩溃时无健康 worker 可分配）：
                # 动态分配给第一个来查询的健康 worker
                w = self._s.workers.get(worker_id)
                if w is not None and w.state == "ALIVE":
                    seq.owner_worker_id = worker_id
                    seq.state = "RECOVERED"
                    self._s.worker_sequences[worker_id].add(seq_id)
                    logger.info(
                        f"Orphaned sequence {seq_id!r} claimed by {worker_id}  "
                        f"epoch={seq.epoch}  frontier={seq.committed_frontier}"
                    )
                    transferred.append({
                        "seq_id":             seq_id,
                        "epoch":              seq.epoch,
                        "committed_frontier": seq.committed_frontier,
                    })

        return {"ok": True, "sequences": transferred}

    # ── Debug / Inspection ───────────────────────────────────────────────────

    def _list_workers(self, p: dict) -> dict:
        now = time.time()
        workers = [
            {
                "worker_id":      w.worker_id,
                "host":           w.host,
                "port":           w.port,
                "state":          w.state,
                "last_heartbeat": w.last_heartbeat,
                "idle_secs":      round(now - w.last_heartbeat, 1),
                "seq_count":      len(self._s.worker_sequences.get(w.worker_id, set())),
            }
            for w in self._s.workers.values()
        ]
        return {"ok": True, "workers": workers}

    def _list_sequences(self, p: dict) -> dict:
        limit = int(p.get("limit", 50))
        seqs  = [s.to_dict() for s in list(self._s.sequences.values())[:limit]]
        return {"ok": True, "total": len(self._s.sequences), "sequences": seqs}

    def _list_blocks(self, p: dict) -> dict:
        limit  = int(p.get("limit", 50))
        blocks = [b.to_dict() for b in list(self._s.blocks.values())[:limit]]
        return {"ok": True, "total": len(self._s.blocks), "blocks": blocks}

    # ── Failure Detection (called by background monitor) ─────────────────────

    def check_failures(self) -> List[str]:
        """
        Scan workers for missed heartbeats.  For each newly-failed worker,
        discard its PENDING blocks and transfer sequence ownership to a
        healthy peer.  Returns the list of newly-failed worker IDs.
        """
        now          = time.time()
        newly_failed = []

        for worker_id, w in self._s.workers.items():
            if w.state == "ALIVE" and (now - w.last_heartbeat) > FAILURE_TIMEOUT:
                w.state = "FAILED"
                newly_failed.append(worker_id)
                idle = now - w.last_heartbeat
                logger.warning(
                    f"Worker FAILED: {worker_id} "
                    f"(no heartbeat for {idle:.1f}s)"
                )
                self._transfer_sequences(worker_id)

        return newly_failed

    def _transfer_sequences(self, failed_worker_id: str) -> None:
        """
        Transfer all active sequences from a failed worker.

        For each sequence:
          1. Discard PENDING blocks (may be incomplete writes)
          2. Increment epoch (fences out any delayed writes from the old worker)
          3. Assign to the least-loaded healthy worker (PENDING_TRANSFER)
        """
        owned = list(self._s.worker_sequences.get(failed_worker_id, set()))
        if not owned:
            return

        healthy = [
            wid for wid, w in self._s.workers.items()
            if w.state == "ALIVE" and wid != failed_worker_id
        ]

        for seq_id in owned:
            seq = self._s.sequences.get(seq_id)
            if seq is None or seq.state not in ("ACTIVE", "RECOVERED"):
                continue

            # Discard all PENDING blocks beyond the committed frontier
            for idx, ch in enumerate(seq.block_order):
                if idx <= seq.committed_frontier:
                    continue
                block = self._s.blocks.get(ch)
                if block is not None and block.state == "PENDING":
                    block.state = "DISCARDED"
                    logger.debug(
                        f"Discarded PENDING block {ch:016x} "
                        f"(seq={seq_id!r}, worker={failed_worker_id})"
                    )

            # Increment epoch to fence out the old worker if it revives
            seq.epoch += 1
            seq.state  = "PENDING_TRANSFER"

            if healthy:
                new_owner = min(
                    healthy,
                    key=lambda wid: len(self._s.worker_sequences.get(wid, set())),
                )
                seq.owner_worker_id = new_owner
                self._s.worker_sequences[new_owner].add(seq_id)
                logger.info(
                    f"Sequence transferred: {seq_id!r}  "
                    f"{failed_worker_id} → {new_owner}  "
                    f"epoch={seq.epoch}  frontier={seq.committed_frontier}"
                )
            else:
                # 没有在线的健康 worker —— 保持 PENDING_TRANSFER 状态、
                # 清空 owner，等下一个注册的 worker 通过 get_pending_transfers
                # 主动认领。不降级为 FAILED，否则序列永久丢失。
                seq.owner_worker_id = ""
                logger.warning(
                    f"Sequence {seq_id!r} is orphaned (no healthy workers).  "
                    f"Will be assigned to the next worker that calls "
                    f"get_pending_transfers."
                )

        self._s.worker_sequences[failed_worker_id].clear()


# ─── Async TCP Server ─────────────────────────────────────────────────────────

class GMSServer:
    """
    Asyncio TCP server.  One persistent connection per MetadataClient.
    Each line received is a JSON request; the handler returns a JSON response line.
    """

    def __init__(self, host: str = GMS_HOST, port: int = GMS_PORT):
        self.host     = host
        self.port     = port
        self._state   = GMSState()
        self._handlers = GMSHandlers(self._state)
        self._client_count = 0

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        peer = writer.get_extra_info("peername")
        self._client_count += 1
        logger.info(f"Client connected: {peer}  (total={self._client_count})")

        try:
            while True:
                line = await reader.readline()
                if not line:
                    break                           # client disconnected

                try:
                    msg    = json.loads(line.decode())
                    req_id = msg.get("id", "")
                    method = msg.get("method", "")
                    params = msg.get("params", {})
                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    result = {"ok": False, "error": f"Parse error: {e}", "id": ""}
                else:
                    result         = self._handlers.dispatch(method, params)
                    result["id"]   = req_id

                response = json.dumps(result) + "\n"
                writer.write(response.encode())
                await writer.drain()

        except (asyncio.IncompleteReadError, ConnectionResetError):
            pass
        finally:
            self._client_count -= 1
            logger.info(f"Client disconnected: {peer}  (total={self._client_count})")
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    async def _monitor_task(self) -> None:
        """Background task: check for worker heartbeat timeouts."""
        while True:
            await asyncio.sleep(MONITOR_INTERVAL)
            try:
                self._handlers.check_failures()
            except Exception:
                logger.exception("Error in failure-detection monitor")

    async def run(self) -> None:
        server = await asyncio.start_server(
            self._handle_client, self.host, self.port
        )
        asyncio.create_task(self._monitor_task())

        addr = server.sockets[0].getsockname()
        logger.info("=" * 55)
        logger.info(f"  GMS server listening on {addr[0]}:{addr[1]}")
        logger.info(f"  HEARTBEAT_INTERVAL = {HEARTBEAT_INTERVAL}s")
        logger.info(f"  FAILURE_TIMEOUT    = {FAILURE_TIMEOUT}s")
        logger.info(f"  MONITOR_INTERVAL   = {MONITOR_INTERVAL}s")
        logger.info("=" * 55)

        async with server:
            await server.serve_forever()


# ─── Entry Point ─────────────────────────────────────────────────────────────

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="GMS-KVCache Global Metadata Service"
    )
    parser.add_argument("--host",  default=GMS_HOST,  help="Bind address")
    parser.add_argument("--port",  default=GMS_PORT,  type=int, help="Listen port")
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG logging")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger("gms").setLevel(logging.DEBUG)

    srv = GMSServer(host=args.host, port=args.port)
    asyncio.run(srv.run())


if __name__ == "__main__":
    main()
