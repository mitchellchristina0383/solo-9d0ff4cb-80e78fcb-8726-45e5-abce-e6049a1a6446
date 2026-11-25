"""End-to-end verification of incremental continuation training.

Builds a finished parent run, appends a headerless increment CSV, creates a
continuation run and trains it, then compares against an uninterrupted
reference fit over the *merged* data (old rows first, new rows after).

Two tail shapes are exercised:

* ``--tail full``    : the parent row count is an exact multiple of the group
  size, so every parent group is reused and training resumes on the first
  increment row;
* ``--tail partial`` : the parent ends with an under-filled group, which is
  rolled back: the continuation restores the parameters/momentum from just
  before that group and re-groups its old rows together with the increment.

Checks printed in the JSON report:

* final w/b/v_w/v_b and every per-group audit value match the from-scratch
  merged fit bit-for-bit (IEEE-754 equality, not tolerance);
* the reused groups produced no new gradient computation in the continuation
  session;
* the audit replay passes;
* peak rows held never exceeds the chunk limit, independent of total rows;
* the parent run stays finished and viewable after the continuation is built.

    python3 scripts/verify_continuation.py --work /tmp/v-cont --tail partial
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from trainer import engine  # noqa: E402


def write_csv(path: str, rows, header: bool = True) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        if header:
            fh.write("measured,reference\n")
        for x, y in rows:
            fh.write(f"{x!r},{y!r}\n")


def parent_rows(n: int):
    return [
        (float(i) * 0.37 - 4.0,
         2.5 * (float(i) * 0.37 - 4.0) - 1.25 + (i % 7) * 0.01)
        for i in range(n)
    ]


def incr_rows(n: int):
    return [(float(i) * 0.13 + 2.0, float(i) * 0.05 + 0.2) for i in range(n)]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--work", required=True)
    p.add_argument("--tail", choices=("full", "partial"), required=True)
    p.add_argument("--group-size", type=int, default=200)
    p.add_argument("--chunk-limit", type=int, default=16)
    p.add_argument("--parent-rows", type=int, default=1000)
    p.add_argument("--incr-rows", type=int, default=537)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--json-out", default=None)
    args = p.parse_args(argv)

    if os.path.exists(args.work):
        shutil.rmtree(args.work)
    os.makedirs(args.work)
    db = os.path.join(args.work, "verify.sqlite3")
    parent_csv = os.path.join(args.work, "parent.csv")
    incr_csv = os.path.join(args.work, "incr.csv")
    merged_csv = os.path.join(args.work, "merged.csv")
    ref_csv = os.path.join(args.work, "reference.csv")

    # Pick a parent row count that produces the requested tail shape.
    g = args.group_size
    if args.tail == "full":
        n_parent = (args.parent_rows // g) * g
    else:
        base = (args.parent_rows // g) * g
        n_parent = max(g + 1, base + (g // 3 or 1))
    n_incr = args.incr_rows

    write_csv(parent_csv, parent_rows(n_parent))
    write_csv(incr_csv, incr_rows(n_incr), header=False)  # headerless increment
    write_csv(ref_csv, parent_rows(n_parent) + incr_rows(n_incr))

    conn = engine.connect(db)
    parent_id = engine.create_run(
        conn, parent_csv, "parent.csv", parent_csv,
        group_size=g, chunk_limit=args.chunk_limit,
        lr=args.lr, momentum=args.momentum, init_w=0.0, init_b=0.0,
    )
    engine.train(conn, parent_id, parent_csv)
    parent_before = dict(engine.get_run(conn, parent_id))

    commits_seen: list[int] = []

    def watch(info):
        if info.get("event") == "committed":
            commits_seen.append(int(info["epoch"]))

    cont_id = engine.create_continuation(
        conn, parent_id, parent_csv, incr_csv, "incr.csv",
        "incr-stored.csv", "merged-stored.csv", merged_csv,
    )
    cont_initial = dict(engine.get_run(conn, cont_id))
    engine.train(conn, cont_id, merged_csv, progress=watch)
    cont = dict(engine.get_run(conn, cont_id))
    parent_after = dict(engine.get_run(conn, parent_id))

    meta = engine.scan_file(ref_csv)
    ref = engine.reference_fit(
        ref_csv, data_start=meta.data_start, total_rows=meta.rows,
        group_size=g, chunk_limit=args.chunk_limit, lr=args.lr,
        momentum=args.momentum, init_w=0.0, init_b=0.0,
    )

    # Fresh fit on the merged data for per-group bit comparison.
    fresh_id = engine.create_run(
        conn, ref_csv, "fresh.csv", ref_csv,
        group_size=g, chunk_limit=max(1, args.chunk_limit // 3),
        lr=args.lr, momentum=args.momentum, init_w=0.0, init_b=0.0,
    )
    engine.train(conn, fresh_id, ref_csv)
    cols = ("idx", "samples", "grad_w", "grad_b", "loss_mean",
            "v_w", "v_b", "w_after", "b_after")
    cont_groups = [dict(r) for r in conn.execute(
        f"SELECT {', '.join(cols)} FROM updates WHERE run_id=? ORDER BY idx",
        (cont_id,)).fetchall()]
    fresh_groups = [dict(r) for r in conn.execute(
        f"SELECT {', '.join(cols)} FROM updates WHERE run_id=? ORDER BY idx",
        (fresh_id,)).fetchall()]
    per_group_exact = cont_groups == fresh_groups

    audit = engine.audit(conn, cont_id, merged_csv)
    reused = cont_initial["reused_updates"]
    keys = ("w", "b", "v_w", "v_b")
    payload = {
        "tail": args.tail,
        "parent_rows": n_parent,
        "increment_rows": n_incr,
        "merged_rows": n_parent + n_incr,
        "group_size": g,
        "chunk_limit": args.chunk_limit,
        "parent_updates": parent_before["updates_done"],
        "reused_updates": reused,
        "reused_samples": cont_initial["reused_samples"],
        "groups_computed_in_continuation_session": sorted(commits_seen),
        "no_gradient_work_for_reused_groups":
            all(idx >= reused for idx in commits_seen),
        "continuation_final": {k: cont[k] for k in
                               ("updates_done", "samples_done", "status",
                                "w", "b", "v_w", "v_b",
                                "peak_rows_held", "peak_rss_kb")},
        "reference_final": {k: ref[k] for k in
                            ("updates", "samples", "w", "b", "v_w", "v_b")},
        "bit_exact_final": all(cont[k] == ref[k] for k in keys),
        "per_group_bit_exact": per_group_exact,
        "group_count_match": len(cont_groups) == len(fresh_groups) == ref["updates"],
        "peak_rows_bounded": cont["peak_rows_held"] <= args.chunk_limit,
        "audit_ok": audit["ok"],
        "audit_problems": audit["problems"],
        "parent_unchanged": all(
            parent_before[k] == parent_after[k]
            for k in ("status", "updates_done", "samples_done", "w", "b",
                      "v_w", "v_b", "total_rows")
        ),
        "seam_padding": cont["seam_padding"],
    }
    payload["identical_to_merged_from_scratch"] = bool(
        payload["bit_exact_final"] and payload["per_group_bit_exact"]
        and payload["group_count_match"] and payload["audit_ok"]
        and payload["no_gradient_work_for_reused_groups"]
        and payload["peak_rows_bounded"] and payload["parent_unchanged"]
        and cont["status"] == "finished"
        and cont["samples_done"] == n_parent + n_incr
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
    conn.close()
    return 0 if payload["identical_to_merged_from_scratch"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
