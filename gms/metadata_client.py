"""
metadata_client.py — MetadataClient for vLLM Instances
=======================================================
Lightweight client library embedded in each vLLM worker.

Responsibilities:
  - Registers the worker with GMS on startup
  - Sends heartbeats every HEARTBEAT_INTERVAL seconds (background thread)
  - Provides the full GMS RPC API for the write/read/failover paths
  - Thread-safe: vLLM is multi-threaded; a single lock serialises socket I/O
  - Auto-reconnects transparently after connection loss

Typical write path (2PC + dedup):
    client = MetadataClient(worker_id="vllm-prefill-0", ...)
    client.start()

    # Before S3 PUT: check if already committed (dedup) + declare intent
    result = client.register_write(
        chunk_hash=0xabcdef, seq_id="req-001", epoch=0, size_bytes=4096
    )
    if not result["ok"]:
        return                          # stale epoch; this worker lost ownership
    if result.get("deduplicated"):
        return                          # already in MinIO; skip the PUT

    # Do the actual S3 PUT here (worker → MinIO directly, GMS not involved)
    s3_client.put_object("kvcache", object_key, kv_bytes)

    # After S3 PUT: confirm success
    client.commit_write(
        chunk_hash=0xabcdef, seq_id="req-001", epoch=0,
        minio_node="localhost:9000", object_key="000000000abcdef0",
        bucket="kvcache", size_bytes=4096,
    )

Failover recovery path:
    # New owner, after receiving a PENDING_TRANSFER notification:
    transfers = client.get_pending_transfers()
    for t in transfers["sequences"]:
        result = client.get_committed_blocks(t["seq_id"])
        for block in result["blocks"]:
            # Load block.object_key from block.minio_node via S3 GET
            kv = s3_get(block["minio_node"], block["bucket"], block["object_key"])
            load_into_gpu(kv)
        # Recompute the one block beyond committed_frontier (at most 1 by invariant)
"""

import json
import logging
import socket
import threading
import time
import uuid
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ─── Tunables ────────────────────────────────────────────────────────────────

HEARTBEAT_INTERVAL  = 5    # seconds between heartbeat RPCs
RECONNECT_DELAY     = 2    # seconds before a reconnect attempt
CONNECT_TIMEOUT     = 3.0  # seconds to wait for TCP connect
RECV_TIMEOUT        = 5.0  # seconds to wait for each server response
MAX_RETRIES         = 2    # how many times to retry a failed RPC


# ─── Exceptions ──────────────────────────────────────────────────────────────

class GMSConnectionError(RuntimeError):
    """Raised when all retries are exhausted and the GMS is unreachable."""


# ─── Client ──────────────────────────────────────────────────────────────────

class MetadataClient:
    """
    Thread-safe GMS client.

    Maintains a single persistent TCP connection to the GMS server.
    On connection loss, the next RPC call transparently reconnects and
    re-registers the worker before retrying.

    All public methods are blocking and return a dict.  Check result["ok"]
    before using other fields.  On persistent GMS unavailability they raise
    GMSConnectionError; callers should fall back to local-only operation.
    """

    def __init__(
        self,
        worker_id:   str,
        gms_host:    str = "localhost",
        gms_port:    int = 8500,
        worker_host: str = "localhost",
        worker_port: int = 8000,
    ) -> None:
        """
        Args:
            worker_id:   Unique ID for this vLLM instance (e.g. "vllm-prefill-0").
            gms_host:    Hostname/IP of the GMS server.
            gms_port:    TCP port of the GMS server (default 8500).
            worker_host: This worker's reachable hostname (reported to GMS for routing).
            worker_port: This worker's vLLM HTTP port (reported to GMS for routing).
        """
        self.worker_id   = worker_id
        self.gms_host    = gms_host
        self.gms_port    = gms_port
        self.worker_host = worker_host
        self.worker_port = worker_port

        self._sock:  Optional[socket.socket] = None
        self._fobj   = None          # makefile("r") wrapper for readline()
        self._lock   = threading.Lock()
        self._running = False
        self._hb_thread: Optional[threading.Thread] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """
        Connect to GMS, register this worker, and start the background
        heartbeat thread.  Must be called before any RPC method.
        """
        self._running = True
        self._connect()
        self._rpc_register_worker()

        self._hb_thread = threading.Thread(
            target=self._heartbeat_loop,
            name=f"gms-hb-{self.worker_id}",
            daemon=True,
        )
        self._hb_thread.start()
        logger.info(
            f"MetadataClient started: worker_id={self.worker_id!r}  "
            f"gms={self.gms_host}:{self.gms_port}"
        )

    def stop(self) -> None:
        """Stop the heartbeat thread and close the connection."""
        self._running = False
        if self._hb_thread is not None:
            self._hb_thread.join(timeout=HEARTBEAT_INTERVAL + 1)
        with self._lock:
            self._close_socket()
        logger.info(f"MetadataClient stopped: {self.worker_id!r}")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()

    # ── Public RPC API ────────────────────────────────────────────────────────

    def register_sequence(self, seq_id: str) -> dict:
        """
        Register a new decode sequence with GMS.
        Returns {"ok": True, "epoch": int, "seq_id": str}.
        Call this once per incoming request before the first register_write.
        """
        return self._rpc("register_sequence", {
            "seq_id":    seq_id,
            "worker_id": self.worker_id,
        })

    def lookup(self, chunk_hashes: List[int]) -> dict:
        """
        Query GMS for block locations.

        Returns:
          {
            "ok":     True,
            "hits":   [{"chunk_hash", "minio_node", "bucket", "object_key",
                         "size_bytes", "state", ...}, ...],
            "misses": [chunk_hash, ...],
          }

        A "hit" means the block is COMMITTED in MinIO at the given location.
        Use the returned minio_node + object_key to issue an S3 GET directly.
        """
        return self._rpc("lookup", {"chunk_hashes": chunk_hashes})

    def register_write(
        self,
        chunk_hash:  int,
        seq_id:      str,
        epoch:       int,
        size_bytes:  int = 0,
    ) -> dict:
        """
        2PC Phase 1: declare intent to write a block to MinIO.

        Returns:
          {"ok": True,  "deduplicated": False}          → proceed with S3 PUT
          {"ok": True,  "deduplicated": True}            → skip S3 PUT (already committed)
          {"ok": False, "reason": "STALE_EPOCH", ...}   → worker lost ownership; stop

        Only call commit_write after the S3 PUT succeeds.
        """
        return self._rpc("register_write", {
            "chunk_hash":  chunk_hash,
            "seq_id":      seq_id,
            "worker_id":   self.worker_id,
            "epoch":       epoch,
            "size_bytes":  size_bytes,
        })

    def commit_write(
        self,
        chunk_hash:  int,
        seq_id:      str,
        epoch:       int,
        minio_node:  str,
        object_key:  str,
        bucket:      str = "kvcache",
        size_bytes:  int = 0,
    ) -> dict:
        """
        2PC Phase 2: confirm S3 PUT succeeded.  Promotes block to COMMITTED
        and advances the committed_frontier for the sequence.

        Returns:
          {"ok": True,  "committed_frontier": int}
          {"ok": False, "reason": "STALE_EPOCH", ...}   → late delivery; stop
        """
        return self._rpc("commit_write", {
            "chunk_hash":  chunk_hash,
            "seq_id":      seq_id,
            "worker_id":   self.worker_id,
            "epoch":       epoch,
            "minio_node":  minio_node,
            "object_key":  object_key,
            "bucket":      bucket,
            "size_bytes":  size_bytes,
        })

    def report_access(self, chunk_hashes: List[int]) -> dict:
        """
        Batched async access reporting (call every ~100ms with recently-read hashes).
        Updates last_accessed and access_count for admission gate statistics.
        """
        return self._rpc("report_access", {
            "chunk_hashes": chunk_hashes,
            "worker_id":    self.worker_id,
        })

    def get_committed_blocks(self, seq_id: str) -> dict:
        """
        Failover recovery: return all COMMITTED blocks for a sequence up to
        the committed_frontier.  Called by the new owner after taking over.

        Returns:
          {
            "ok":                 True,
            "seq_id":             str,
            "epoch":              int,
            "committed_frontier": int,
            "blocks": [
              {"index", "chunk_hash", "minio_node", "bucket",
               "object_key", "size_bytes"},
              ...
            ]
          }

        Iterate blocks in order and issue S3 GETs to rebuild the GPU cache.
        Recompute any blocks beyond committed_frontier (at most 1 by the invariant).
        """
        return self._rpc("get_committed_blocks", {"seq_id": seq_id})

    def get_pending_transfers(self) -> dict:
        """
        Poll GMS for sequences newly transferred to this worker by a failover.
        Returns each sequence exactly once; subsequent calls will not repeat it.

        Returns:
          {
            "ok": True,
            "sequences": [
              {"seq_id", "epoch", "committed_frontier"},
              ...
            ]
          }

        For each returned sequence, call get_committed_blocks(seq_id) and
        reload the cache from MinIO.
        """
        return self._rpc("get_pending_transfers", {"worker_id": self.worker_id})

    # ── Debug helpers ─────────────────────────────────────────────────────────

    def list_workers(self) -> dict:
        """Return all registered workers and their liveness status."""
        return self._rpc("list_workers", {})

    def list_sequences(self, limit: int = 50) -> dict:
        """Return registered sequences (for debugging)."""
        return self._rpc("list_sequences", {"limit": limit})

    def list_blocks(self, limit: int = 50) -> dict:
        """Return registered blocks (for debugging)."""
        return self._rpc("list_blocks", {"limit": limit})

    # ── Internal: Connection & Transport ─────────────────────────────────────

    def _connect(self) -> None:
        """Open TCP connection to GMS.  Caller must hold self._lock or be in startup."""
        self._close_socket()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(CONNECT_TIMEOUT)
        sock.connect((self.gms_host, self.gms_port))
        sock.settimeout(RECV_TIMEOUT)
        # TCP_NODELAY: disable Nagle — we send small JSON messages and need low latency
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._sock = sock
        self._fobj = sock.makefile("r", encoding="utf-8")
        logger.debug(f"Connected to GMS at {self.gms_host}:{self.gms_port}")

    def _close_socket(self) -> None:
        if self._fobj is not None:
            try:
                self._fobj.close()
            except Exception:
                pass
            self._fobj = None
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None

    def _rpc_register_worker(self) -> dict:
        """Internal: register worker without going through the retry loop."""
        return self._rpc("register_worker", {
            "worker_id": self.worker_id,
            "host":      self.worker_host,
            "port":      self.worker_port,
        })

    def _rpc(self, method: str, params: dict) -> dict:
        """
        Send one JSON-RPC request and return the parsed response.
        Thread-safe via self._lock.
        Transparently reconnects (and re-registers) on connection failure,
        up to MAX_RETRIES attempts.
        """
        req_id   = uuid.uuid4().hex[:8]
        payload  = json.dumps({"id": req_id, "method": method, "params": params}) + "\n"
        payload_b = payload.encode()

        with self._lock:
            for attempt in range(MAX_RETRIES + 1):
                try:
                    if self._sock is None:
                        self._connect()
                        if method != "register_worker":
                            # Re-register before replaying the original call
                            self._rpc_register_worker()

                    self._sock.sendall(payload_b)
                    line = self._fobj.readline()
                    if not line:
                        raise GMSConnectionError("Server closed connection")
                    resp = json.loads(line)
                    return resp

                except (OSError, GMSConnectionError, json.JSONDecodeError) as exc:
                    logger.warning(
                        f"RPC {method!r} attempt {attempt + 1}/{MAX_RETRIES + 1} failed: {exc}"
                    )
                    self._close_socket()

                    if attempt < MAX_RETRIES:
                        time.sleep(RECONNECT_DELAY)
                    else:
                        raise GMSConnectionError(
                            f"GMS RPC {method!r} failed after "
                            f"{MAX_RETRIES + 1} attempts"
                        ) from exc

    # ── Heartbeat loop (background thread) ───────────────────────────────────

    def _heartbeat_loop(self) -> None:
        """Sends a heartbeat to GMS every HEARTBEAT_INTERVAL seconds."""
        while self._running:
            time.sleep(HEARTBEAT_INTERVAL)
            if not self._running:
                break
            try:
                self._rpc("heartbeat", {"worker_id": self.worker_id})
                logger.debug(f"Heartbeat sent: {self.worker_id!r}")
            except GMSConnectionError as e:
                logger.warning(f"Heartbeat failed (GMS unreachable): {e}")
            except Exception as e:
                logger.warning(f"Heartbeat unexpected error: {e}")
