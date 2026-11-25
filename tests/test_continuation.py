"""Tests for incremental continuation runs.

Covers both tail shapes required by the feature:

* the parent's last group is full  -> every parent group is reused, the new
  run starts reading at the first increment row;
* the parent's last group is short -> that group is discarded, the new run
  restores the parameters/momentum from just before it and re-groups the old
  partial rows together with the increment.

In every case the continuation must reproduce a from-scratch fit on the merged
data bit-for-bit (final params/momentum *and* every per-group audit value),
must not recompute gradients for reused groups, and must keep the chunk-sized
memory bound regardless of the total sample count.
"""

from __future__ import annotations

import io
import os
import shutil
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import create_app  # noqa: E402
from trainer import engine  # noqa: E402


def make_rows(n: int, seed: float = 0.0) -> list[tuple[float, float]]:
    return [
        (float(i) * 0.37 - 4.0 + seed,
         2.5 * (float(i) * 0.37 - 4.0) - 1.25 + (i % 7) * 0.01)
        for i in range(n)
    ]


def write_csv(path: str, rows: list[tuple[float, float]], header: bool = True,
              blanks_every: int = 0) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        if header:
            fh.write("measured,reference\n")
        for i, (x, y) in enumerate(rows):
            fh.write(f"{x!r},{y!r}\n")
            if blanks_every and i % blanks_every == blanks_every - 1:
                fh.write("\n")


class ContinuationBase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="continuation-test-")
        self.db = os.path.join(self.tmp, "t.sqlite3")
        self.parent_csv = os.path.join(self.tmp, "parent.csv")
        self.incr_csv = os.path.join(self.tmp, "incr.csv")
        self.merged_csv = os.path.join(self.tmp, "merged.csv")
        self.conn = engine.connect(self.db)

    def tearDown(self) -> None:
        self.conn.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def make_parent(self, n: int, *, group_size: int = 8, chunk_limit: int = 3,
                    **kw) -> int:
        defaults = dict(lr=0.01, momentum=0.9, init_w=0.0, init_b=0.0)
        defaults.update(kw)
        write_csv(self.parent_csv, make_rows(n))
        rid = engine.create_run(
            self.conn, self.parent_csv, "parent.csv", self.parent_csv,
            group_size=group_size, chunk_limit=chunk_limit, **defaults,
        )
        engine.train(self.conn, rid, self.parent_csv)
        return rid

    def make_increment(self, n: int, *, header: bool = False) -> list[tuple[float, float]]:
        inc = [(float(i) * 0.13 + 2.0, float(i) * 0.05 + 0.2) for i in range(n)]
        write_csv(self.incr_csv, inc, header=header)
        return inc

    def make_continuation(self, parent_id: int) -> int:
        return engine.create_continuation(
            self.conn, parent_id, self.parent_csv, self.incr_csv,
            "incr.csv", "incr-stored.csv", "merged-stored.csv", self.merged_csv,
        )

    def merged_reference(self, parent_rows, incr_rows, *, group_size: int = 8,
                         **kw) -> dict:
        ref_path = os.path.join(self.tmp, "ref.csv")
        write_csv(ref_path, parent_rows + incr_rows)
        meta = engine.scan_file(ref_path)
        defaults = dict(lr=0.01, momentum=0.9, init_w=0.0, init_b=0.0,
                        chunk_limit=5)
        defaults.update(kw)
        return engine.reference_fit(
            ref_path, data_start=meta.data_start,
            total_rows=len(parent_rows) + len(incr_rows),
            group_size=group_size, **defaults,
        )

    def assert_bit_exact_groups(self, cont_id: int, fresh_id: int) -> None:
        cols = ("idx", "samples", "grad_w", "grad_b", "loss_mean",
                "v_w", "v_b", "w_after", "b_after")
        got = [dict(r) for r in self.conn.execute(
            f"SELECT {', '.join(cols)} FROM updates WHERE run_id=? ORDER BY idx",
            (cont_id,)).fetchall()]
        want = [dict(r) for r in self.conn.execute(
            f"SELECT {', '.join(cols)} FROM updates WHERE run_id=? ORDER BY idx",
            (fresh_id,)).fetchall()]
        self.assertEqual(len(got), len(want))
        self.assertEqual(got, want)


class TestFullTailContinuation(ContinuationBase):
    def test_reuses_every_group_and_skips_their_gradients(self):
        parent_id = self.make_parent(24, group_size=8)  # exactly 3 full groups
        inc = self.make_increment(10)
        cont_id = self.make_continuation(parent_id)
        row = engine.get_run(self.conn, cont_id)
        self.assertEqual((row["reused_updates"], row["reused_samples"]), (3, 24))
        self.assertEqual((row["updates_done"], row["samples_done"]), (3, 24))
        self.assertEqual(row["part_rows"], 0)
        # resumes at the parent's terminal parameters/momentum, not at init
        parent = engine.get_run(self.conn, parent_id)
        self.assertEqual((row["w"], row["b"], row["v_w"], row["v_b"]),
                         (parent["w"], parent["b"], parent["v_w"], parent["v_b"]))

        # Train while watching which groups actually get computed: groups 0..2
        # must never appear as committed in this session.
        seen_commits: list[int] = []
        engine.train(
            self.conn, cont_id, self.merged_csv,
            progress=lambda info: seen_commits.append(info["epoch"])
            if info.get("event") == "committed" else None,
        )
        self.assertEqual(sorted(seen_commits), [3, 4])  # only the two new groups

        row = engine.get_run(self.conn, cont_id)
        self.assertEqual((row["status"], row["updates_done"], row["samples_done"]),
                         ("finished", 5, 34))
        ref = self.merged_reference(make_rows(24), inc)
        for key in ("w", "b", "v_w", "v_b"):
            self.assertEqual(row[key], ref[key], key)
        self.assertTrue(
            engine.audit(self.conn, cont_id, self.merged_csv)["ok"]
        )

        # A fresh fit on the merged data must produce identical per-group values.
        ref_path = os.path.join(self.tmp, "ref.csv")
        write_csv(ref_path, make_rows(24) + inc)
        fresh_id = engine.create_run(
            self.conn, ref_path, "fresh.csv", ref_path,
            group_size=8, chunk_limit=2, lr=0.01, momentum=0.9,
            init_w=0.0, init_b=0.0,
        )
        engine.train(self.conn, fresh_id, ref_path)
        self.assert_bit_exact_groups(cont_id, fresh_id)

    def test_seam_newline_when_parent_has_no_trailing_newline(self):
        # Parent data ending without a newline forces a seam padding byte.
        with open(self.parent_csv, "wb") as fh:
            fh.write(b"x,y\n1.0,2.0\n3.0,4.0")  # no trailing newline
        rid = engine.create_run(
            self.conn, self.parent_csv, "p.csv", self.parent_csv,
            group_size=2, chunk_limit=2, lr=0.01, momentum=0.9,
            init_w=0.0, init_b=0.0,
        )
        engine.train(self.conn, rid, self.parent_csv)
        with open(self.incr_csv, "wb") as fh:
            fh.write(b"5.0,6.0\n7.0,8.0\n")
        cid = engine.create_continuation(
            self.conn, rid, self.parent_csv, self.incr_csv,
            "incr.csv", "i.csv", "m.csv", self.merged_csv,
        )
        row = engine.get_run(self.conn, cid)
        self.assertEqual(row["seam_padding"], 1)
        engine.train(self.conn, cid, self.merged_csv)
        row = engine.get_run(self.conn, cid)
        self.assertEqual(row["status"], "finished")
        # the fused-line accident never happened: all four rows were counted
        self.assertEqual(row["samples_done"], 4)
        self.assertTrue(engine.audit(self.conn, cid, self.merged_csv)["ok"])

    def test_increment_with_bom_is_accepted(self):
        self.make_parent(16, group_size=8)
        with open(self.incr_csv, "wb") as fh:
            fh.write(b"\xef\xbb\xbf1.0,2.0\n3.0,4.0\n")
        cid = self.make_continuation(self.conn.execute(
            "SELECT id FROM runs ORDER BY id DESC LIMIT 1").fetchone()["id"])
        self.assertEqual(engine.get_run(self.conn, cid)["incr_rows"], 2)


class TestPartialTailContinuation(ContinuationBase):
    def test_short_final_group_restarts_before_it(self):
        parent_id = self.make_parent(21, group_size=8)  # 2 full groups + 5 rows
        parent = engine.get_run(self.conn, parent_id)
        self.assertEqual((parent["updates_done"], parent["part_rows"]), (3, 0))
        inc = self.make_increment(7)
        cont_id = self.make_continuation(parent_id)
        row = engine.get_run(self.conn, cont_id)
        # only the two complete groups are reused; the 5-row tail is rolled back
        self.assertEqual((row["reused_updates"], row["reused_samples"]), (2, 16))
        self.assertEqual((row["updates_done"], row["samples_done"]), (2, 16))
        self.assertEqual(row["part_rows"], 0)
        checkpoint = self.conn.execute(
            "SELECT w_after, b_after, v_w, v_b FROM updates "
            "WHERE run_id=? AND idx=1", (parent_id,)).fetchone()
        self.assertEqual((row["w"], row["b"], row["v_w"], row["v_b"]),
                         (checkpoint["w_after"], checkpoint["b_after"],
                          checkpoint["v_w"], checkpoint["v_b"]))
        # read offset points at the first row of the old partial group
        second = self.conn.execute(
            "SELECT offset_after FROM updates WHERE run_id=? AND idx=1",
            (parent_id,)).fetchone()
        expected_off = second["offset_after"] - int(parent["data_start"])
        self.assertEqual(row["next_offset"], expected_off)
        self.assertEqual(row["part_start_offset"], expected_off)

        # all groups from idx 2 onward are recomputed (old tail + increment)
        recomputed: list[int] = []
        engine.train(
            self.conn, cont_id, self.merged_csv,
            progress=lambda info: recomputed.append(info["epoch"])
            if info.get("event") == "committed" else None,
        )
        self.assertEqual(sorted(recomputed), [2, 3])
        row = engine.get_run(self.conn, cont_id)
        self.assertEqual((row["status"], row["updates_done"], row["samples_done"]),
                         ("finished", 4, 28))
        ref = self.merged_reference(make_rows(21), inc)
        for key in ("w", "b", "v_w", "v_b"):
            self.assertEqual(row[key], ref[key], key)
        self.assertTrue(
            engine.audit(self.conn, cont_id, self.merged_csv)["ok"]
        )

        ref_path = os.path.join(self.tmp, "ref.csv")
        write_csv(ref_path, make_rows(21) + inc)
        fresh_id = engine.create_run(
            self.conn, ref_path, "fresh.csv", ref_path,
            group_size=8, chunk_limit=1, lr=0.01, momentum=0.9,
            init_w=0.0, init_b=0.0,
        )
        engine.train(self.conn, fresh_id, ref_path)
        self.assert_bit_exact_groups(cont_id, fresh_id)

    def test_parent_smaller_than_one_group_restarts_from_init(self):
        parent_id = self.make_parent(5, group_size=8)  # one short group, no full one
        self.make_increment(6)
        cont_id = self.make_continuation(parent_id)
        row = engine.get_run(self.conn, cont_id)
        self.assertEqual((row["reused_updates"], row["reused_samples"]), (0, 0))
        self.assertEqual((row["w"], row["b"], row["v_w"], row["v_b"]),
                         (0.0, 0.0, 0.0, 0.0))
        self.assertEqual(row["next_offset"], 0)
        engine.train(self.conn, cont_id, self.merged_csv)
        row = engine.get_run(self.conn, cont_id)
        inc = [(float(i) * 0.13 + 2.0, float(i) * 0.05 + 0.2) for i in range(6)]
        ref = self.merged_reference(make_rows(5), inc)
        for key in ("w", "b", "v_w", "v_b"):
            self.assertEqual(row[key], ref[key], key)

    def test_blank_lines_in_parent_and_increment_keep_row_order(self):
        write_csv(self.parent_csv, make_rows(21), blanks_every=5)
        rid = engine.create_run(
            self.conn, self.parent_csv, "p.csv", self.parent_csv,
            group_size=8, chunk_limit=3, lr=0.01, momentum=0.9,
            init_w=0.0, init_b=0.0,
        )
        engine.train(self.conn, rid, self.parent_csv)
        inc = self.make_increment(9)
        # add blank lines inside the increment too
        with open(self.incr_csv, "a", encoding="utf-8") as fh:
            fh.write("\n\n")
        cid = self.make_continuation(rid)
        engine.train(self.conn, cid, self.merged_csv)
        row = engine.get_run(self.conn, cid)
        self.assertEqual(row["samples_done"], 30)
        ref = self.merged_reference(make_rows(21), inc)
        for key in ("w", "b", "v_w", "v_b"):
            self.assertEqual(row[key], ref[key], key)
        self.assertTrue(engine.audit(self.conn, cid, self.merged_csv)["ok"])

    def test_second_batch_forms_a_bit_exact_linear_chain(self):
        # A third batch arrives after the first continuation: chaining another
        # continuation must stay bit-exact versus a single from-scratch fit.
        parent_id = self.make_parent(21, group_size=8)  # partial tail
        inc1 = self.make_increment(7)
        c1 = self.make_continuation(parent_id)
        engine.train(self.conn, c1, self.merged_csv)
        merged1 = self.merged_csv

        inc2 = [(float(i) * 0.09 - 1.0, float(i) * 0.03 + 0.4) for i in range(9)]
        incr2_csv = os.path.join(self.tmp, "incr2.csv")
        merged2_csv = os.path.join(self.tmp, "merged2.csv")
        write_csv(incr2_csv, inc2, header=False)
        c2 = engine.create_continuation(
            self.conn, c1, merged1, incr2_csv, "incr2.csv",
            "incr2-stored.csv", "merged2-stored.csv", merged2_csv,
        )
        engine.train(self.conn, c2, merged2_csv)
        row = engine.get_run(self.conn, c2)
        self.assertEqual(row["parent_run_id"], c1)
        self.assertEqual(row["samples_done"], 21 + 7 + 9)

        ref_path = os.path.join(self.tmp, "ref-chain.csv")
        write_csv(ref_path, make_rows(21) + inc1 + inc2)
        meta = engine.scan_file(ref_path)
        ref = engine.reference_fit(
            ref_path, data_start=meta.data_start, total_rows=meta.rows,
            group_size=8, chunk_limit=2, lr=0.01, momentum=0.9,
            init_w=0.0, init_b=0.0,
        )
        for key in ("w", "b", "v_w", "v_b"):
            self.assertEqual(row[key], ref[key], key)
        self.assertTrue(engine.audit(self.conn, c2, merged2_csv)["ok"])


class TestContinuationValidation(ContinuationBase):
    def test_header_in_increment_is_rejected(self):
        parent_id = self.make_parent(24)
        self.make_increment(4, header=True)
        with self.assertRaisesRegex(engine.InputError, "表头"):
            self.make_continuation(parent_id)
        self.assertFalse(os.path.exists(self.merged_csv))
        # no new run appeared, parent stays viewable
        self.assertEqual(
            engine.get_run(self.conn, parent_id)["status"], "finished"
        )

    def test_bad_rows_are_rejected(self):
        parent_id = self.make_parent(24)
        with open(self.incr_csv, "w") as fh:
            fh.write("1.0,2.0\n3.0\n")
        with self.assertRaisesRegex(engine.InputError, "2 列"):
            self.make_continuation(parent_id)
        with open(self.incr_csv, "w") as fh:
            fh.write("nan,2.0\n1.0,1.0\n")
        with self.assertRaises(engine.InputError):
            self.make_continuation(parent_id)

    def test_unfinished_parent_rejected(self):
        write_csv(self.parent_csv, make_rows(40))
        rid = engine.create_run(
            self.conn, self.parent_csv, "p.csv", self.parent_csv,
            group_size=8, chunk_limit=3, lr=0.01, momentum=0.9,
            init_w=0.0, init_b=0.0,
        )
        engine.train(self.conn, rid, self.parent_csv, max_updates=1)
        self.make_increment(3)
        with self.assertRaisesRegex(engine.InputError, "尚未完成"):
            self.make_continuation(rid)

    def test_only_one_continuation_per_parent(self):
        parent_id = self.make_parent(24)
        self.make_increment(4)
        first = self.make_continuation(parent_id)
        self.assertTrue(first)
        with self.assertRaisesRegex(engine.InputError, "已创建过续训"):
            self.make_continuation(parent_id)

    def test_changed_parent_file_rejected(self):
        parent_id = self.make_parent(24)
        self.make_increment(4)
        with open(self.parent_csv, "a") as fh:
            fh.write("9.0,9.0\n")
        with self.assertRaisesRegex(engine.InputError, "数据文件已变化"):
            self.make_continuation(parent_id)

    def test_failure_leaves_no_run_and_cleans_files(self):
        parent_id = self.make_parent(24)
        # bad increment -> everything must roll back
        with open(self.incr_csv, "w") as fh:
            fh.write("oops,not-a-number\n")
        with self.assertRaises(engine.InputError):
            self.make_continuation(parent_id)
        rows = self.conn.execute("SELECT COUNT(*) AS n FROM runs").fetchone()
        self.assertEqual(rows["n"], 1)  # only the parent remains
        self.assertFalse(os.path.exists(self.merged_csv))


class TestContinuationMemoryAndChunks(ContinuationBase):
    def test_peak_residency_independent_of_total_rows(self):
        # Large parent + large increment, tiny chunk limit: peak held rows must
        # stay bounded by the chunk limit even though the merged set is huge.
        n_parent, n_incr, group, chunk = 4000, 2000, 64, 5
        parent_id = self.make_parent(n_parent, group_size=group,
                                     chunk_limit=chunk, lr=1e-6)
        self.make_increment(n_incr)
        cont_id = self.make_continuation(parent_id)
        engine.train(self.conn, cont_id, self.merged_csv)
        row = engine.get_run(self.conn, cont_id)
        self.assertEqual(row["samples_done"], n_parent + n_incr)
        self.assertLessEqual(row["peak_rows_held"], chunk)
        # reused groups mean the new session only read the increment-sized tail
        ref = self.merged_reference(
            make_rows(n_parent),
            [(float(i) * 0.13 + 2.0, float(i) * 0.05 + 0.2)
             for i in range(n_incr)],
            group_size=group, lr=1e-6,
        )
        for key in ("w", "b", "v_w", "v_b"):
            self.assertEqual(row[key], ref[key], key)

    def test_pause_and_resume_inside_continuation(self):
        parent_id = self.make_parent(21, group_size=8)
        self.make_increment(11)
        cont_id = self.make_continuation(parent_id)
        engine.set_pause_after_groups(self.conn, cont_id, 1)
        r1 = engine.train(self.conn, cont_id, self.merged_csv)
        self.assertEqual(r1.status, "paused")
        engine.set_pause_after_groups(self.conn, cont_id, None)
        engine.set_chunk_limit(self.conn, cont_id, 1)
        engine.train(self.conn, cont_id, self.merged_csv)
        row = engine.get_run(self.conn, cont_id)
        inc = [(float(i) * 0.13 + 2.0, float(i) * 0.05 + 0.2) for i in range(11)]
        ref = self.merged_reference(make_rows(21), inc)
        for key in ("w", "b", "v_w", "v_b"):
            self.assertEqual(row[key], ref[key], key)
        self.assertTrue(engine.audit(self.conn, cont_id, self.merged_csv)["ok"])


class TestContinuationWebFlow(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="continuation-web-")
        self.app = create_app({
            "DB_PATH": os.path.join(self.tmp, "db.sqlite3"),
            "UPLOAD_DIR": os.path.join(self.tmp, "uploads"),
            "SECRET_KEY": "test",
            "TESTING": True,
        })
        self.client = self.app.test_client()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _wait_idle(self, run_id: int, timeout: float = 60.0) -> dict:
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self.app.test_client() as c:
                body = c.get(f"/api/runs/{run_id}").get_json()
            if not body["worker_running"] and body["run"]["status"] != "running":
                return body
            time.sleep(0.05)
        raise AssertionError("worker did not finish")

    def _upload_parent(self, n: int, group_size: int) -> int:
        csv_text = "x,y\n" + "".join(
            f"{x:.6f},{y:.6f}\n" for x, y in make_rows(n))
        resp = self.client.post("/upload", data={
            "csv": (io.BytesIO(csv_text.encode()), "parent.csv"),
            "group_size": str(group_size), "chunk_limit": "3",
            "lr": "0.01", "momentum": "0.9", "init_w": "0", "init_b": "0",
        }, content_type="multipart/form-data")
        self.assertEqual(resp.status_code, 302)
        with engine.session(self.app.config["DB_PATH"]) as conn:
            run_id = engine.list_runs(conn)[0]["id"]
        self.client.post(f"/runs/{run_id}/start", data={"pause_after": ""})
        body = self._wait_idle(run_id)
        self.assertEqual(body["run"]["status"], "finished")
        return run_id

    @staticmethod
    def _six_f(path: str, rows: list[tuple[float, float]]) -> None:
        with open(path, "w") as fh:
            fh.write("x,y\n")
            for x, y in rows:
                fh.write(f"{x:.6f},{y:.6f}\n")

    def test_web_continuation_full_and_partial(self):
        for tag, n, group in (("full", 24, 8), ("partial", 21, 8)):
            parent_id = self._upload_parent(n, group)
            inc = [(float(i) * 0.13 + 2.0, float(i) * 0.05 + 0.2)
                   for i in range(9)]
            inc_text = "".join(f"{x:.6f},{y:.6f}\n" for x, y in inc)
            resp = self.client.post(
                f"/runs/{parent_id}/continue",
                data={"incr_csv": (io.BytesIO(inc_text.encode()), "incr.csv")},
                content_type="multipart/form-data",
            )
            self.assertEqual(resp.status_code, 302)
            with engine.session(self.app.config["DB_PATH"]) as conn:
                cont = conn.execute(
                    "SELECT * FROM runs WHERE parent_run_id=?", (parent_id,)
                ).fetchone()
            self.assertIsNotNone(cont, f"{tag}: continuation run was not created")
            cont_id = cont["id"]
            # detail page shows lineage and the parent page stays available
            page = self.client.get(f"/runs/{cont_id}").get_data(as_text=True)
            self.assertIn(f"#{parent_id}", page)
            parent_page = self.client.get(f"/runs/{parent_id}").get_data(
                as_text=True)
            self.assertIn("增量续训", parent_page)

            self.client.post(f"/runs/{cont_id}/start", data={"pause_after": ""})
            body = self._wait_idle(cont_id)
            self.assertEqual(body["run"]["status"], "finished")
            audit = self.client.get(
                f"/api/runs/{cont_id}/audit").get_json()
            self.assertTrue(audit["ok"], audit["problems"])

            merged_path = engine.resolve_data_path(
                self.app.config["UPLOAD_DIR"], body["run"]["stored_name"])
            # reference uses the exact same decimal text as the uploaded files
            ref_path = os.path.join(self.tmp, f"ref-{tag}.csv")
            self._six_f(ref_path, make_rows(n) + inc)
            ref = engine.reference_fit(
                ref_path, data_start=engine.scan_file(ref_path).data_start,
                total_rows=n + 9, group_size=group, chunk_limit=4,
                lr=0.01, momentum=0.9, init_w=0.0, init_b=0.0,
            )
            for key in ("w", "b", "v_w", "v_b"):
                self.assertEqual(body["run"][key], ref[key], f"{tag}:{key}")
            self.assertTrue(os.path.exists(merged_path))

    def test_web_bad_increment_shows_reason_and_keeps_parent(self):
        parent_id = self._upload_parent(24, 8)
        resp = self.client.post(
            f"/runs/{parent_id}/continue",
            data={"incr_csv": (io.BytesIO(b"x,y\n1,2\n3,4\n"), "bad.csv")},
            content_type="multipart/form-data",
            follow_redirects=True,
        )
        page = resp.get_data(as_text=True)
        self.assertIn("表头", page)
        with engine.session(self.app.config["DB_PATH"]) as conn:
            runs = engine.list_runs(conn)
        self.assertEqual(len(runs), 1)  # parent only
        self.assertEqual(runs[0]["status"], "finished")
        # no stray files left in the upload directory
        files = os.listdir(self.app.config["UPLOAD_DIR"])
        self.assertEqual(len(files), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
