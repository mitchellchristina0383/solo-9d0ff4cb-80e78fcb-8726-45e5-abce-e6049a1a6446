"""Tests for the streaming trainer: grouping, memory bound, pause and crash resume."""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from trainer import engine  # noqa: E402


def write_csv(path: str, rows: list[tuple[float, float]], header: bool = True,
              blanks_every: int = 0) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        if header:
            fh.write("measured,reference\n")
        for i, (x, y) in enumerate(rows):
            fh.write(f"{x!r},{y!r}\n")
            if blanks_every and i % blanks_every == blanks_every - 1:
                fh.write("\n")


def make_rows(n: int) -> list[tuple[float, float]]:
    return [(float(i) * 0.37 - 4.0, 2.5 * (float(i) * 0.37 - 4.0) - 1.25 + (i % 7) * 0.01)
            for i in range(n)]


class CrashAfter(Exception):
    """Simulated abrupt process exit."""


class Base(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="trainer-test-")
        self.db = os.path.join(self.tmp, "t.sqlite3")
        self.csv = os.path.join(self.tmp, "data.csv")
        self.conn = engine.connect(self.db)

    def tearDown(self) -> None:
        self.conn.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    # -- helpers ----------------------------------------------------------
    def create(self, **kw) -> int:
        defaults = dict(group_size=10, chunk_limit=4, lr=0.01, momentum=0.9,
                        init_w=0.0, init_b=0.0)
        defaults.update(kw)
        return engine.create_run(
            self.conn, self.csv, "data.csv", self.csv, **defaults
        )

    def reference(self, **kw) -> dict:
        defaults = dict(group_size=10, chunk_limit=4, lr=0.01, momentum=0.9,
                        init_w=0.0, init_b=0.0)
        defaults.update(kw)
        meta = engine.scan_file(self.csv)
        return engine.reference_fit(
            self.csv, data_start=meta.data_start, total_rows=meta.rows, **defaults
        )

    def run_row(self, run_id: int):
        return engine.get_run(self.conn, run_id)


class TestGrouping(Base):
    def test_groups_and_final_partial_group(self):
        write_csv(self.csv, make_rows(25))
        rid = self.create(group_size=10)
        result = engine.train(self.conn, rid, self.csv)
        self.assertEqual(result.status, "finished")
        self.assertEqual(result.updates, 3)
        self.assertEqual(result.samples, 25)
        counts = [r["samples"] for r in self.conn.execute(
            "SELECT samples FROM updates WHERE run_id=? ORDER BY idx", (rid,))]
        self.assertEqual(counts, [10, 10, 5])
        self.assertTrue(engine.audit(self.conn, rid)["ok"])

    def test_sample_counting_skips_blank_lines(self):
        write_csv(self.csv, make_rows(23), blanks_every=5)
        rid = self.create(group_size=7)
        engine.train(self.conn, rid, self.csv)
        meta = engine.scan_file(self.csv)
        self.assertEqual(meta.rows, 23)
        counts = [r["samples"] for r in self.conn.execute(
            "SELECT samples FROM updates WHERE run_id=? ORDER BY idx", (rid,))]
        self.assertEqual(counts, [7, 7, 7, 2])
        self.assertEqual(self.run_row(rid)["samples_done"], 23)

    def test_headerless_file(self):
        write_csv(self.csv, make_rows(12), header=False)
        meta = engine.scan_file(self.csv)
        self.assertFalse(meta.has_header)
        self.assertEqual(meta.rows, 12)
        rid = self.create(group_size=5)
        engine.train(self.conn, rid, self.csv)
        self.assertEqual(self.run_row(rid)["samples_done"], 12)

    def test_invalid_rows_rejected(self):
        with open(self.csv, "w", encoding="utf-8") as fh:
            fh.write("x,y\n1,2\n3\n")
        with self.assertRaises(engine.InputError):
            engine.create_run(self.conn, self.csv, "bad.csv", self.csv,
                              group_size=2, chunk_limit=1, lr=0.1, momentum=0.0,
                              init_w=0.0, init_b=0.0)

    def test_nan_rejected(self):
        with open(self.csv, "w", encoding="utf-8") as fh:
            fh.write("x,y\n1,2\nnan,3\n")
        with self.assertRaises(engine.InputError):
            engine.scan_file(self.csv)

    def test_single_row_and_huge_group(self):
        write_csv(self.csv, [(3.0, 7.0)])
        rid = self.create(group_size=1000)
        engine.train(self.conn, rid, self.csv)
        row = self.run_row(rid)
        self.assertEqual((row["updates_done"], row["samples_done"]), (1, 1))


class TestMemoryBound(Base):
    def test_chunks_never_exceed_limit(self):
        write_csv(self.csv, make_rows(1000))
        rid = self.create(group_size=100, chunk_limit=7)
        seen = []

        def observe(info):
            if info["event"] == "chunk":
                seen.append(info["rows_in_chunk"])

        engine.train(self.conn, rid, self.csv, progress=observe)
        self.assertTrue(seen)
        self.assertLessEqual(max(seen), 7)
        self.assertEqual(self.run_row(rid)["peak_rows_held"], 7)
        # each group is read 100/7 -> 14 full chunks + a 2-row tail
        self.assertEqual(seen.count(2), 10)

    def test_chunk_limit_larger_than_group(self):
        write_csv(self.csv, make_rows(30))
        rid = self.create(group_size=10, chunk_limit=1000)
        engine.train(self.conn, rid, self.csv)
        self.assertEqual(self.run_row(rid)["peak_rows_held"], 10)

    def test_chunk_size_does_not_change_result(self):
        write_csv(self.csv, make_rows(103))
        rid = self.create(group_size=10, chunk_limit=3)
        engine.train(self.conn, rid, self.csv)
        got = self.run_row(rid)
        ref = self.reference(group_size=10, chunk_limit=64)
        self.assertEqual(got["w"], ref["w"])
        self.assertEqual(got["b"], ref["b"])

    def test_peak_rss_tracked(self):
        write_csv(self.csv, make_rows(200))
        rid = self.create(group_size=20, chunk_limit=5)
        engine.train(self.conn, rid, self.csv)
        self.assertGreater(self.run_row(rid)["peak_rss_kb"], 0)


class TestUninterrupted(Base):
    def test_matches_reference_exactly(self):
        write_csv(self.csv, make_rows(57))
        rid = self.create(group_size=8, chunk_limit=3, lr=0.02, momentum=0.8)
        engine.train(self.conn, rid, self.csv)
        ref = self.reference(group_size=8, chunk_limit=1, lr=0.02, momentum=0.8)
        row = self.run_row(rid)
        self.assertEqual((row["updates_done"], row["samples_done"]), (8, 57))
        for key in ("w", "b", "v_w", "v_b"):
            self.assertEqual(row[key], ref[key], key)
        audit = engine.audit(self.conn, rid)
        self.assertTrue(audit["ok"], audit["problems"])

    def test_zero_momentum_is_plain_sgd(self):
        write_csv(self.csv, [(1.0, 1.0), (2.0, 2.0)])
        rid = self.create(group_size=2, chunk_limit=2, lr=1.0, momentum=0.0)
        engine.train(self.conn, rid, self.csv)
        row = self.run_row(rid)
        # one group of both rows from (0, 0): g_w = -5, g_b = -3, lr = 1
        self.assertAlmostEqual(row["w"], 5.0, places=12)
        self.assertAlmostEqual(row["b"], 3.0, places=12)


class TestPauseResume(Base):
    def test_pause_after_n_chunks_resumes_at_next_row(self):
        write_csv(self.csv, make_rows(95))
        rid = self.create(group_size=10, chunk_limit=3)
        engine.set_pause_after_chunks(self.conn, rid, 2)
        r1 = engine.train(self.conn, rid, self.csv)
        self.assertEqual(r1.status, "paused")
        self.assertEqual(r1.chunks_this_session, 2)
        row = self.run_row(rid)
        # two 3-row chunks fold into a *partial* group: no update yet
        self.assertEqual((row["updates_done"], row["samples_done"]), (0, 0))
        self.assertEqual(row["part_rows"], 6)
        offset1 = row["next_offset"]
        self.assertGreater(offset1, row["data_start"])
        r2 = engine.train(self.conn, rid, self.csv)
        self.assertEqual(r2.chunks_this_session, 2)
        self.assertEqual(self.run_row(rid)["next_offset"] > offset1, True)
        engine.set_pause_after_chunks(self.conn, rid, None)
        r3 = engine.train(self.conn, rid, self.csv)
        self.assertEqual((r3.status, r3.updates, r3.samples), ("finished", 10, 95))
        ref = self.reference(group_size=10)
        self.assertEqual((r3.w, r3.b, r3.v_w, r3.v_b),
                         (ref["w"], ref["b"], ref["v_w"], ref["v_b"]))
        audit = engine.audit(self.conn, rid)
        self.assertTrue(audit["ok"], audit["problems"])
        # rows are never counted twice: part_rows were carried across sessions
        self.assertEqual(self.run_row(rid)["part_rows"], 0)

    def test_pause_after_n_groups_budget(self):
        write_csv(self.csv, make_rows(95))
        rid = self.create(group_size=10, chunk_limit=3)
        engine.set_pause_after_groups(self.conn, rid, 3)
        r1 = engine.train(self.conn, rid, self.csv)
        self.assertEqual((r1.status, r1.updates, r1.samples, r1.part_rows),
                         ("paused", 3, 30, 0))
        engine.set_pause_after_groups(self.conn, rid, None)
        r2 = engine.train(self.conn, rid, self.csv)
        self.assertEqual((r2.status, r2.updates, r2.samples), ("finished", 10, 95))
        ref = self.reference(group_size=10, chunk_limit=3)
        self.assertEqual((r2.w, r2.b, r2.v_w, r2.v_b),
                         (ref["w"], ref["b"], ref["v_w"], ref["v_b"]))

    def test_partial_group_pause_never_rereads_rows(self):
        write_csv(self.csv, make_rows(40))
        rid = self.create(group_size=10, chunk_limit=3)
        engine.set_pause_after_chunks(self.conn, rid, 1)
        seen_offsets = []
        while True:
            row = self.run_row(rid)
            seen_offsets.append(row["next_offset"])
            result = engine.train(self.conn, rid, self.csv)
            if result.status == "finished":
                break
        # each session advanced the read offset: no row was consumed twice
        self.assertEqual(seen_offsets, sorted(set(seen_offsets)))
        row = self.run_row(rid)
        self.assertEqual((row["updates_done"], row["samples_done"]), (4, 40))
        ref = self.reference(group_size=10, chunk_limit=3)
        self.assertEqual((row["w"], row["b"]), (ref["w"], ref["b"]))

    def test_repeated_pause_cycles_equal_uninterrupted(self):
        write_csv(self.csv, make_rows(211))
        rid = self.create(group_size=13, chunk_limit=5)
        engine.set_pause_after_chunks(self.conn, rid, 2)
        for _ in range(100):
            result = engine.train(self.conn, rid, self.csv)
            if result.status == "finished":
                break
        row = self.run_row(rid)
        ref = self.reference(group_size=13, chunk_limit=5)
        self.assertEqual(row["status"], "finished")
        self.assertEqual(row["updates_done"], 17)
        for key in ("w", "b", "v_w", "v_b"):
            self.assertEqual(row[key], ref[key], key)

    def test_chunk_limit_can_shrink_between_segments(self):
        write_csv(self.csv, make_rows(120))
        rid = self.create(group_size=10, chunk_limit=6)
        engine.set_pause_after_chunks(self.conn, rid, 4)
        engine.train(self.conn, rid, self.csv)
        engine.set_pause_after_chunks(self.conn, rid, None)
        engine.set_chunk_limit(self.conn, rid, 2)
        self.assertEqual(self.run_row(rid)["chunk_limit"], 2)
        engine.set_chunk_limit(self.conn, rid, 1)
        engine.train(self.conn, rid, self.csv)
        row = self.run_row(rid)
        self.assertEqual(row["updates_done"], 12)
        ref = self.reference(group_size=10, chunk_limit=6)
        self.assertEqual(row["w"], ref["w"])
        self.assertEqual(row["peak_rows_held"], 6)

    def test_active_pause_request_is_honoured(self):
        write_csv(self.csv, make_rows(500))
        rid = self.create(group_size=10, chunk_limit=3)
        calls = {"n": 0}

        def observe(info):
            if info["event"] == "committed":
                calls["n"] += 1
                if calls["n"] == 2:
                    engine.request_pause(self.conn, rid)

        result = engine.train(self.conn, rid, self.csv, progress=observe)
        self.assertEqual(result.status, "paused")
        self.assertEqual(self.run_row(rid)["updates_done"], 2)
        self.assertEqual(self.run_row(rid)["pause_requested"], 0)


class TestCrashResume(Base):
    def _crash_at(self, rid: int, stage: str, epoch: int, chunk: int = 0) -> None:
        count = {"chunks": 0}

        def hook(stage_name, info):
            if stage_name == "chunk" and stage == "chunk":
                if info["epoch"] == epoch:
                    if count["chunks"] == chunk:
                        raise CrashAfter(f"simulated exit in group {epoch} chunk {chunk}")
                    count["chunks"] += 1
            elif stage_name == stage and info.get("epoch") == epoch:
                raise CrashAfter(f"simulated exit {stage} group {epoch}")

        with self.assertRaises(CrashAfter):
            engine.train(self.conn, rid, self.csv, crash_hook=hook)

    def test_crash_mid_group_rereads_at_most_one_group(self):
        write_csv(self.csv, make_rows(97))
        rid = self.create(group_size=10, chunk_limit=4)
        self._crash_at(rid, "chunk", epoch=3, chunk=1)
        row = self.run_row(rid)
        self.assertEqual((row["updates_done"], row["samples_done"]), (3, 30))
        self.assertEqual(row["status"], "crashed")
        # restart recovery, then resume
        self.assertEqual(engine.recover(self.conn), 0)
        engine.train(self.conn, rid, self.csv)
        ref = self.reference(group_size=10, chunk_limit=4)
        row = self.run_row(rid)
        self.assertEqual((row["updates_done"], row["samples_done"]), (10, 97))
        for key in ("w", "b", "v_w", "v_b"):
            self.assertEqual(row[key], ref[key], key)
        self.assertTrue(engine.audit(self.conn, rid)["ok"])

    def test_crash_before_commit_does_not_lose_committed_groups(self):
        write_csv(self.csv, make_rows(50))
        rid = self.create(group_size=10, chunk_limit=2)
        self._crash_at(rid, "before_commit", epoch=2)
        row = self.run_row(rid)
        self.assertEqual((row["updates_done"], row["samples_done"]), (2, 20))
        self.assertEqual(row["next_offset"], self._offset_after_rows(20))
        engine.recover(self.conn)
        engine.train(self.conn, rid, self.csv)
        ref = self.reference(group_size=10)
        row = self.run_row(rid)
        self.assertEqual(row["updates_done"], 5)
        self.assertEqual((row["w"], row["b"]), (ref["w"], ref["b"]))

    def test_crash_after_commit_keeps_that_group(self):
        write_csv(self.csv, make_rows(50))
        rid = self.create(group_size=10, chunk_limit=2)
        self._crash_at(rid, "after_commit", epoch=3)
        row = self.run_row(rid)
        self.assertEqual((row["updates_done"], row["samples_done"]), (4, 40))
        self.assertEqual(row["status"], "crashed")
        engine.recover(self.conn)
        engine.train(self.conn, rid, self.csv)
        ref = self.reference(group_size=10)
        row = self.run_row(rid)
        self.assertEqual(row["updates_done"], 5)
        self.assertEqual((row["w"], row["b"], row["v_w"], row["v_b"]),
                         (ref["w"], ref["b"], ref["v_w"], ref["v_b"]))

    def test_hard_process_kill_then_resume(self):
        """Kill a real child process mid-training, then resume in this process."""
        import subprocess
        import textwrap

        write_csv(self.csv, make_rows(400))
        rid = self.create(group_size=10, chunk_limit=3)
        script = textwrap.dedent(f"""
            import sys
            sys.path.insert(0, {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))!r})
            from trainer import engine
            conn = engine.connect({self.db!r})
            def hook(stage, info):
                if stage == 'chunk' and info['epoch'] == 7 and info['rows_this_group'] >= 9:
                    import os; os._exit(9)
            engine.train(conn, {rid}, {self.csv!r}, crash_hook=hook)
        """)
        path = os.path.join(self.tmp, "child.py")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(script)
        proc = subprocess.run([sys.executable, "-I", path], cwd=self.tmp,
                              capture_output=True, text=True, timeout=120)
        self.assertEqual(proc.returncode, 9, proc.stderr)
        self.conn.close()
        self.conn = engine.connect(self.db)
        row = self.run_row(rid)
        self.assertEqual(row["status"], "running")
        self.assertEqual(row["updates_done"], 7)
        self.assertEqual(engine.recover(self.conn), 1)
        self.assertEqual(self.run_row(rid)["status"], "crashed")
        engine.train(self.conn, rid, self.csv)
        ref = self.reference(group_size=10, chunk_limit=3)
        row = self.run_row(rid)
        self.assertEqual((row["updates_done"], row["samples_done"]), (40, 400))
        for key in ("w", "b", "v_w", "v_b"):
            self.assertEqual(row[key], ref[key], key)
        self.assertTrue(engine.audit(self.conn, rid)["ok"])

    def _offset_after_rows(self, n: int) -> int:
        meta = engine.scan_file(self.csv)
        with open(self.csv, "rb") as fh:
            fh.seek(meta.data_start)
            for _ in range(n):
                fh.readline()
            return fh.tell()


class TestDivergence(Base):
    def test_divergence_fails_cleanly(self):
        # inverse-variance blow-up: w oscillates and overflows within a few groups
        write_csv(self.csv, [(10.0 + i * 1e-9, 1.0) for i in range(8)])
        rid = self.create(group_size=1, chunk_limit=1, lr=1e300, momentum=0.999)
        with self.assertRaises(engine.InputError):
            engine.train(self.conn, rid, self.csv)
        row = self.run_row(rid)
        self.assertEqual(row["status"], "failed")
        self.assertIn("发散", row["message"])
        # the diverging group was never committed; earlier groups stay intact
        self.assertEqual(row["updates_done"], 1)
        self.assertEqual(row["part_rows"], 0)
        self.assertTrue(engine.audit(self.conn, rid)["ok"])
        # and the run refuses to proceed rather than writing NaN/NULL
        with self.assertRaises(engine.StateError):
            engine.train(self.conn, rid, self.csv)


class TestSafety(Base):
    def test_data_change_rejected(self):
        write_csv(self.csv, make_rows(20))
        rid = self.create(group_size=5)
        engine.train(self.conn, rid, self.csv, max_updates=1)
        with open(self.csv, "a", encoding="utf-8") as fh:
            fh.write("1,1\n")
        with self.assertRaises(engine.InputError):
            engine.train(self.conn, rid, self.csv)

    def test_duplicate_open_run_rejected(self):
        write_csv(self.csv, make_rows(20))
        self.create()
        with self.assertRaises(engine.InputError):
            self.create()

    def test_chunk_limit_cannot_change_while_running(self):
        write_csv(self.csv, make_rows(20))
        rid = self.create()
        with engine.transaction(self.conn):
            self.conn.execute("UPDATE runs SET status='running' WHERE id=?", (rid,))
        with self.assertRaises(engine.StateError):
            engine.set_chunk_limit(self.conn, rid, 5)

    def test_validation(self):
        for kwargs in (
            dict(group_size=0), dict(chunk_limit=0),
            dict(lr=0.0), dict(lr=float("nan")), dict(momentum=1.0),
            dict(momentum=-0.1),
        ):
            defaults = dict(group_size=5, chunk_limit=2, lr=0.1,
                            momentum=0.5, init_w=0.0, init_b=0.0)
            defaults.update(kwargs)
            with self.assertRaises(engine.InputError):
                engine.validate_params(
                    defaults["group_size"], defaults["chunk_limit"],
                    defaults["lr"], defaults["momentum"],
                    defaults["init_w"], defaults["init_b"], None,
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
