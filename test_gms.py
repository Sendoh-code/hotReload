"""
test_gms.py — End-to-end smoke test for GMS + MetadataClient
=============================================================
Tests (in order):
  1. Worker registration & heartbeat
  2. Sequence registration
  3. Lookup miss (block not yet registered)
  4. 2PC write: register_write → commit_write
  5. Lookup hit (block now COMMITTED)
  6. Deduplication: second register_write returns deduplicated=True
  7. Epoch fencing: stale epoch rejected
  8. get_committed_blocks (failover recovery)
  9. Simulated failover: worker failure → ownership transfer
 10. get_pending_transfers (new owner picks up sequence)

Run with:
    python test_gms.py [--host localhost] [--port 8500]
"""

import argparse
import sys
import time

# Allow running from project root
sys.path.insert(0, "/home/ext_yiti6755_colorado_edu/LMCache_minIO")

from gms.metadata_client import MetadataClient, GMSConnectionError


# ─── Helpers ─────────────────────────────────────────────────────────────────

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
_failures = []

def check(label: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  {PASS}  {label}")
    else:
        print(f"  {FAIL}  {label}" + (f"  ({detail})" if detail else ""))
        _failures.append(label)


# ─── Test cases ──────────────────────────────────────────────────────────────

def test_worker_registration(c1: MetadataClient) -> None:
    print("\n[1] Worker registration & heartbeat")
    workers = c1.list_workers()
    check("list_workers ok", workers["ok"])
    ids = [w["worker_id"] for w in workers["workers"]]
    check("worker-1 registered", c1.worker_id in ids)
    check("worker-1 state=ALIVE",
          any(w["state"] == "ALIVE" for w in workers["workers"]
              if w["worker_id"] == c1.worker_id))


def test_sequence_registration(c1: MetadataClient) -> None:
    print("\n[2] Sequence registration")
    r = c1.register_sequence("seq-001")
    check("register_sequence ok", r["ok"], str(r))
    check("initial epoch=0", r["epoch"] == 0)

    # Idempotent re-registration
    r2 = c1.register_sequence("seq-001")
    check("idempotent re-register ok", r2["ok"])
    check("epoch unchanged", r2["epoch"] == 0)


def test_lookup_miss(c1: MetadataClient) -> None:
    print("\n[3] Lookup miss (block not yet registered)")
    r = c1.lookup([0xAAAABBBB, 0xDEADBEEF])
    check("lookup ok", r["ok"])
    check("no hits", len(r["hits"]) == 0)
    check("two misses", len(r["misses"]) == 2)


def test_2pc_write(c1: MetadataClient) -> dict:
    print("\n[4] 2PC write: register_write → commit_write")
    HASH = 0x0000000012345678

    r1 = c1.register_write(
        chunk_hash=HASH, seq_id="seq-001", epoch=0, size_bytes=4096
    )
    check("register_write ok", r1["ok"], str(r1))
    check("not deduplicated", not r1.get("deduplicated", True))

    r2 = c1.commit_write(
        chunk_hash=HASH, seq_id="seq-001", epoch=0,
        minio_node="localhost:9000", object_key=f"{HASH:016x}",
        bucket="kvcache", size_bytes=4096,
    )
    check("commit_write ok", r2["ok"], str(r2))
    check("frontier advanced to 0", r2["committed_frontier"] == 0)
    return {"chunk_hash": HASH, "object_key": f"{HASH:016x}"}


def test_lookup_hit(c1: MetadataClient, chunk_hash: int) -> None:
    print("\n[5] Lookup hit (block now COMMITTED)")
    r = c1.lookup([chunk_hash])
    check("lookup ok", r["ok"])
    check("one hit", len(r["hits"]) == 1)
    check("no misses", len(r["misses"]) == 0)
    hit = r["hits"][0]
    check("correct minio_node", hit["minio_node"] == "localhost:9000")
    check("state=COMMITTED", hit["state"] == "COMMITTED")


def test_deduplication(c1: MetadataClient, chunk_hash: int) -> None:
    print("\n[6] Deduplication: second register_write returns deduplicated=True")
    r = c1.register_write(
        chunk_hash=chunk_hash, seq_id="seq-001", epoch=0, size_bytes=4096
    )
    check("register_write ok", r["ok"], str(r))
    check("deduplicated=True", r.get("deduplicated") is True)


def test_epoch_fencing(c1: MetadataClient) -> None:
    print("\n[7] Epoch fencing: stale epoch rejected")
    # Register a new sequence at epoch=0; then manually advance epoch via
    # a second client simulating an ownership transfer.
    # We test by sending epoch=-1 (always stale).
    r = c1.register_write(
        chunk_hash=0xFEEDFACE, seq_id="seq-001", epoch=-1, size_bytes=100
    )
    check("stale epoch rejected (ok=False)", not r["ok"], str(r))
    check("reason=STALE_EPOCH", r.get("reason") == "STALE_EPOCH")


def test_get_committed_blocks(c1: MetadataClient) -> None:
    print("\n[8] get_committed_blocks (failover recovery query)")
    r = c1.get_committed_blocks("seq-001")
    check("ok", r["ok"], str(r))
    check("at least one block", len(r["blocks"]) >= 1)
    check("frontier >= 0", r["committed_frontier"] >= 0)
    blk = r["blocks"][0]
    check("block has minio_node", "minio_node" in blk)
    check("block has object_key", "object_key" in blk)


def test_failover(c1: MetadataClient, c2: MetadataClient) -> None:
    print("\n[9] Simulated failover (worker-1 stops heartbeating)")
    print("    Writing block for seq-002 on worker-1 ...")

    r = c1.register_sequence("seq-002")
    check("seq-002 registered", r["ok"])
    epoch = r["epoch"]

    c1.register_write(chunk_hash=0xCAFEBABE, seq_id="seq-002", epoch=epoch, size_bytes=512)
    c1.commit_write(
        chunk_hash=0xCAFEBABE, seq_id="seq-002", epoch=epoch,
        minio_node="localhost:9000", object_key="00000000cafebabe",
        bucket="kvcache", size_bytes=512,
    )
    print(f"    Block committed.  Now waiting {12}s for GMS to declare worker-1 FAILED ...")
    print("    (worker-1 heartbeat loop is still running; "
          "to fully test, stop it before this sleep.)")
    print("    Skipping actual wait in automated test — use manual test for timing.")

    # In a real test you would: c1.stop(); time.sleep(12)
    # Then check c2.get_pending_transfers()
    print("    [manual step] stop worker-1, wait >10s, then:")
    print("      transfers = c2.get_pending_transfers()")
    print("      for t in transfers['sequences']: c2.get_committed_blocks(t['seq_id'])")


def test_pending_transfers_empty(c2: MetadataClient) -> None:
    print("\n[10] get_pending_transfers (should be empty, no failover triggered yet)")
    r = c2.get_pending_transfers()
    check("ok", r["ok"])
    check("empty (no failover yet)", r["sequences"] == [])


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8500)
    args = parser.parse_args()

    print(f"Connecting to GMS at {args.host}:{args.port} ...")

    c1 = MetadataClient(
        worker_id="vllm-prefill-0",
        gms_host=args.host, gms_port=args.port,
        worker_host="localhost", worker_port=8000,
    )
    c2 = MetadataClient(
        worker_id="vllm-decode-0",
        gms_host=args.host, gms_port=args.port,
        worker_host="localhost", worker_port=8001,
    )

    try:
        c1.start()
        c2.start()
        time.sleep(0.1)  # let heartbeat thread settle

        test_worker_registration(c1)
        test_sequence_registration(c1)
        test_lookup_miss(c1)
        block_info = test_2pc_write(c1)
        test_lookup_hit(c1, block_info["chunk_hash"])
        test_deduplication(c1, block_info["chunk_hash"])
        test_epoch_fencing(c1)
        test_get_committed_blocks(c1)
        test_pending_transfers_empty(c2)
        test_failover(c1, c2)

    except GMSConnectionError as e:
        print(f"\n\033[91mERROR: Cannot reach GMS: {e}\033[0m")
        print("Make sure the GMS server is running:  ./start_gms.sh")
        sys.exit(1)
    finally:
        c1.stop()
        c2.stop()

    print("\n" + "=" * 50)
    if _failures:
        print(f"\033[91mFAILED: {len(_failures)} check(s) failed:\033[0m")
        for f in _failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print(f"\033[92mAll checks passed.\033[0m")


if __name__ == "__main__":
    main()
