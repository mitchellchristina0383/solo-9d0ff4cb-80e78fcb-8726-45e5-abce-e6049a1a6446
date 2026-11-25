"""Command-line entry points: a single training run and a reference comparison.

Used by the demo/verification scripts; the web app calls the same engine.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from trainer import engine


def _int_or_none(value: str):
    value = value.strip()
    return None if value == "" else int(value)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m app.cli", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="创建（可选）并推进一个训练")
    t.add_argument("--db", required=True)
    t.add_argument("--data", required=True)
    t.add_argument("--run-id", type=int, default=None, help="续训已有训练")
    t.add_argument("--name", default=None)
    t.add_argument("--group-size", type=int, default=1000)
    t.add_argument("--chunk-limit", type=int, default=200)
    t.add_argument("--lr", type=float, default=0.001)
    t.add_argument("--momentum", type=float, default=0.9)
    t.add_argument("--init-w", type=float, default=0.0)
    t.add_argument("--init-b", type=float, default=0.0)
    t.add_argument("--pause-after", type=_int_or_none, default=None,
                   help="读完这么多段（chunk）后暂停；留空跑完")
    t.add_argument("--set-chunk-limit", type=int, default=None,
                   help="续训前修改读取上限")
    t.add_argument("--max-updates", type=int, default=None)
    t.add_argument("--json", action="store_true")

    r = sub.add_parser("reference", help="不落盘、不中断的参考训练")
    r.add_argument("--data", required=True)
    r.add_argument("--group-size", type=int, default=1000)
    r.add_argument("--chunk-limit", type=int, default=200)
    r.add_argument("--lr", type=float, default=0.001)
    r.add_argument("--momentum", type=float, default=0.9)
    r.add_argument("--init-w", type=float, default=0.0)
    r.add_argument("--init-b", type=float, default=0.0)

    a = sub.add_parser("audit", help="复核训练记录与参数/动量/进度的一致性")
    a.add_argument("--db", required=True)
    a.add_argument("--run-id", type=int, required=True)

    a2 = sub.add_parser("show", help="打印训练状态")
    a2.add_argument("--db", required=True)
    a2.add_argument("--run-id", type=int, required=True)

    args = p.parse_args(argv)

    if args.cmd == "reference":
        meta = engine.scan_file(args.data)
        out = engine.reference_fit(
            args.data, data_start=meta.data_start, total_rows=meta.rows,
            group_size=args.group_size, chunk_limit=args.chunk_limit,
            lr=args.lr, momentum=args.momentum,
            init_w=args.init_w, init_b=args.init_b,
        )
        print(json.dumps(out, indent=2))
        return 0

    conn = engine.connect(args.db)
    try:
        if args.cmd == "audit":
            result = engine.audit(conn, args.run_id)
            result["run"] = dict(result["run"])
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0 if result["ok"] else 1
        if args.cmd == "show":
            print(json.dumps(dict(engine.get_run(conn, args.run_id)),
                             indent=2, ensure_ascii=False))
            return 0
        if args.cmd == "train":
            run_id = args.run_id
            if run_id is None:
                name = args.name or os.path.basename(args.data)
                run_id = engine.create_run(
                    conn, args.data, name, args.data,
                    group_size=args.group_size, chunk_limit=args.chunk_limit,
                    lr=args.lr, momentum=args.momentum,
                    init_w=args.init_w, init_b=args.init_b,
                    pause_after=args.pause_after,
                )
            elif args.set_chunk_limit is not None:
                engine.set_chunk_limit(conn, run_id, args.set_chunk_limit)
            elif args.pause_after is not None:
                engine.set_pause_after_chunks(conn, run_id, args.pause_after)
            result = engine.train(conn, run_id, args.data,
                                  max_updates=args.max_updates)
            payload = {
                "run_id": run_id,
                "status": result.status,
                "message": result.message,
                "updates": result.updates,
                "samples": result.samples,
                "total_rows": result.total_rows,
                "w": result.w, "b": result.b,
                "v_w": result.v_w, "v_b": result.v_b,
            }
            print(json.dumps(payload, indent=2))
            return 0
    finally:
        conn.close()
    return 2


if __name__ == "__main__":
    sys.exit(main())
