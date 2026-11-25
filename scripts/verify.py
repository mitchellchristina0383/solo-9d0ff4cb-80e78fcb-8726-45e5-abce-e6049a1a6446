"""Manual verification: both interruption modes versus one uninterrupted run.

    python scripts/verify.py --mode pause --data /tmp/cal.csv --work /tmp/v-pause
    python scripts/verify.py --mode kill  --data /tmp/cal.csv --work /tmp/v-kill

Each mode drives the real trainer in a child process:

* ``pause`` -- the parent configures "pause after N read chunks", so the child
  stops at a chunk boundary *inside* a group.  The partial group accumulator and
  the read offset are committed; the parent resumes and continues at the very
  next row, so no row is ever read twice.
* ``kill``  -- the child calls ``os._exit(9)`` mid-group: no cleanup at all.
  ``recover`` marks the run crashed, the read limit is shrunk, and the run
  re-reads only the incomplete group.

Both end with a comparison against ``reference_fit`` (no DB, no pause) over the
same data: parameters and momentum must match bit-for-bit, the counted samples
must equal the row count, and the audit replay must pass.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from trainer import engine  # noqa: E402

CHILD = r"""
import os, sys
sys.path.insert(0, {root!r})
from trainer import engine

run_id, data, mode, group, chunks = ({run_id!r}, {data!r}, {mode!r},
                                     {group_index!r}, {chunks!r})
seen = {{"n": 0}}

def hook(stage, info):
    if mode != 'kill' or stage != 'chunk' or info['epoch'] != group:
        return
    seen['n'] += 1
    if seen['n'] >= chunks:
        os._exit(9)          # abrupt death: no cleanup, no status update

conn = engine.connect({db!r})
engine.train(conn, run_id, data, crash_hook=hook)
"""


def row_offset(path: str, data_start: int, n: int) -> int:
    """Byte offset just past the n-th data row (blank lines skipped)."""
    with open(path, "rb") as fh:
        fh.seek(data_start)
        seen = 0
        offset = data_start
        while seen < n:
            raw = fh.readline()
            if not raw:
                break
            offset = fh.tell()
            if engine.decode_line(raw).strip():
                seen += 1
        return offset


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["none", "pause", "kill"], required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--work", required=True)
    p.add_argument("--group-size", type=int, default=2000)
    p.add_argument("--chunk-limit", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.0002)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--interrupt-group", type=int, default=2,
                   help="0-based group index where the child is interrupted")
    p.add_argument("--stop-after-chunks", type=int, default=2,
                   help="interrupt after this many chunks of the target group")
    p.add_argument("--json-out", default=None)
    args = p.parse_args(argv)

    if os.path.exists(args.work):
        shutil.rmtree(args.work)
    os.makedirs(args.work)
    db = os.path.join(args.work, "verify.sqlite3")
    conn = engine.connect(db)
    meta = engine.scan_file(args.data)
    run_id = engine.create_run(
        conn, args.data, os.path.basename(args.data), args.data,
        group_size=args.group_size, chunk_limit=args.chunk_limit,
        lr=args.lr, momentum=args.momentum, init_w=0.0, init_b=0.0,
    )
    chunks_per_group = math.ceil(args.group_size / args.chunk_limit)
    if args.mode == "pause":
        # Land the pause boundary in the middle of the chosen group.
        engine.set_pause_after_chunks(
            conn, run_id,
            args.interrupt_group * chunks_per_group + args.stop_after_chunks,
        )
    conn.close()

    def snapshot() -> dict:
        with engine.session(db) as c:
            row = engine.get_run(c, run_id)
        return {k: row[k] for k in (
            "status", "w", "b", "v_w", "v_b", "updates_done", "samples_done",
            "part_rows", "next_offset", "chunk_limit", "peak_rows_held",
            "peak_rss_kb", "chunks_read",
        )}

    interrupted = None
    if args.mode != "none":
        child_path = os.path.join(args.work, "child.py")
        with open(child_path, "w", encoding="utf-8") as fh:
            fh.write(CHILD.format(root=ROOT, run_id=run_id, data=args.data, db=db,
                                  mode=args.mode, group_index=args.interrupt_group,
                                  chunks=args.stop_after_chunks))
        proc = subprocess.run([sys.executable, "-I", child_path], cwd=args.work,
                              timeout=600)
        expected = 0 if args.mode == "pause" else 9
        if proc.returncode != expected:
            raise SystemExit(f"child exited {proc.returncode}, expected {expected}")
        interrupted = snapshot()
        if args.mode == "kill":
            assert interrupted["status"] == "running", interrupted
            assert interrupted["part_rows"] == 0, interrupted  # partial group lost

    conn = engine.connect(db)
    recovered = engine.recover(conn)
    if args.mode == "pause":
        # the pause target is a per-session budget: clear it to run to the end
        engine.set_pause_after_chunks(conn, run_id, None)
    if args.mode == "kill":
        # The last commit was a group boundary: counted samples are a multiple of
        # the group size and the read offset is exactly past them.
        expected_offset = row_offset(args.data, meta.data_start,
                                     interrupted["samples_done"])
        assert interrupted["next_offset"] == expected_offset, interrupted
        engine.set_chunk_limit(conn, run_id, 1)   # shrink the limit on restart
    engine.train(conn, run_id, args.data)
    finished = snapshot()
    audit = engine.audit(conn, run_id)

    ref = engine.reference_fit(
        args.data, data_start=meta.data_start, total_rows=meta.rows,
        group_size=args.group_size, chunk_limit=args.chunk_limit,
        lr=args.lr, momentum=args.momentum, init_w=0.0, init_b=0.0,
    )

    checks = {}
    if args.mode == "pause" and interrupted:
        # deliberate pause: counted samples + partial rows == rows actually read
        read_rows = interrupted["samples_done"] + interrupted["part_rows"]
        checks["pause_persisted_partial_group"] = interrupted["part_rows"] > 0
        checks["pause_offset_matches_rows_read_once"] = (
            interrupted["next_offset"] == row_offset(args.data, meta.data_start, read_rows)
        )
        # after resuming, the row count consumed by the resumed session must be
        # exactly total - rows already read (nothing re-read, nothing skipped)
        checks["pause_resume_consumed_only_remaining"] = (
            finished["samples_done"] == meta.rows and finished["part_rows"] == 0
        )
    if args.mode == "kill" and interrupted:
        checks["kill_rolls_back_only_incomplete_group"] = (
            interrupted["part_rows"] == 0
            and interrupted["samples_done"] % args.group_size == 0
        )

    keys = ("w", "b", "v_w", "v_b")
    diffs = {k: abs(finished[k] - ref[k]) for k in keys}
    exact = all(finished[k] == ref[k] for k in keys)
    payload = {
        "mode": args.mode,
        "rows": meta.rows,
        "group_size": args.group_size,
        "total_groups": ref["updates"],
        "chunk_limit_start": args.chunk_limit,
        "interrupted": interrupted,
        "recovered_rows_marked": recovered,
        "finished": finished,
        "reference": ref,
        "max_abs_diff": diffs,
        "bit_exact": exact,
        "audit_ok": audit["ok"],
        "audit_problems": audit["problems"],
        "checks": checks,
        "identical_to_uninterrupted": bool(
            exact
            and all(checks.values())
            and finished["updates_done"] == ref["updates"]
            and finished["samples_done"] == ref["samples"] == meta.rows
            and audit["ok"]
        ),
    }
    print(json.dumps(payload, indent=2))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
    conn.close()
    return 0 if payload["identical_to_uninterrupted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
