"""End-to-end tests of the web flow via Flask's test client."""

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

CSV = "measured,reference\n" + "".join(
    f"{i * 0.5 - 20:.4f},{1.5 * (i * 0.5 - 20) + 0.75:.4f}\n" for i in range(64)
)


def wait_idle(app, run_id: int, timeout: float = 60.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with app.test_client() as client:
            body = client.get(f"/api/runs/{run_id}").get_json()
        if not body["worker_running"] and body["run"]["status"] != "running":
            return body
        time.sleep(0.05)
    raise AssertionError("worker did not finish in time")


class TestWebFlow(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="trainer-web-")
        self.app = create_app({
            "DB_PATH": os.path.join(self.tmp, "db.sqlite3"),
            "UPLOAD_DIR": os.path.join(self.tmp, "uploads"),
            "SECRET_KEY": "test",
            "TESTING": True,
        })
        self.client = self.app.test_client()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def upload(self, **form):
        data = {
            "csv": (io.BytesIO(CSV.encode()), "cal.csv"),
            "group_size": "8",
            "chunk_limit": "3",
            "lr": "0.02",
            "momentum": "0.9",
            "init_w": "0",
            "init_b": "0",
        }
        data.update(form)
        return self.client.post("/upload", data=data, content_type="multipart/form-data",
                                follow_redirects=False)

    def test_upload_then_pause_and_finish(self):
        resp = self.upload()
        self.assertEqual(resp.status_code, 302)
        with engine.session(self.app.config["DB_PATH"]) as conn:
            run = engine.list_runs(conn)[0]
        run_id = run["id"]
        self.assertEqual((run["status"], run["total_rows"]), ("ready", 64))

        page = self.client.get(f"/runs/{run_id}")
        self.assertIn("训练 #", page.get_data(as_text=True))

        # pause after 2 groups, resume with a smaller chunk limit, finish
        self.client.post(f"/runs/{run_id}/start",
                         data={"pause_after": "2", "chunk_limit": "3"})
        body = wait_idle(self.app, run_id)
        self.assertEqual(body["run"]["status"], "paused")
        # two 3-row chunks: a partial group is committed to disk
        self.assertEqual(body["run"]["updates_done"], 0)
        self.assertEqual(body["run"]["samples_done"], 0)
        self.assertEqual(body["run"]["part_rows"], 6)
        self.assertEqual(body["run"]["next_offset"] > body["run"]["data_start"], True)

        self.client.post(f"/runs/{run_id}/chunk-limit", data={"chunk_limit": "1"})
        self.client.post(f"/runs/{run_id}/start",
                         data={"pause_after": "", "chunk_limit": "1"})
        body = wait_idle(self.app, run_id)
        self.assertEqual(body["run"]["status"], "finished")
        self.assertEqual(body["run"]["updates_done"], 8)
        self.assertEqual(body["run"]["samples_done"], 64)
        self.assertEqual(body["run"]["chunk_limit"], 1)

        audit = self.client.get(f"/api/runs/{run_id}/audit").get_json()
        self.assertTrue(audit["ok"], audit["problems"])

        ref = engine.reference_fit(
            os.path.join(self.app.config["UPLOAD_DIR"], body["run"]["stored_name"]),
            data_start=body["run"]["data_start"], total_rows=64, group_size=8,
            chunk_limit=5, lr=0.02, momentum=0.9, init_w=0.0, init_b=0.0,
        )
        for key in ("w", "b", "v_w", "v_b"):
            self.assertEqual(body["run"][key], ref[key], key)

    def test_pause_button_during_run(self):
        big = "x,y\n" + "".join(f"{i * 0.001},{2.0 * i * 0.001}\n" for i in range(120000))
        self.upload(csv=(io.BytesIO(big.encode()), "big.csv"),
                    group_size="500", chunk_limit="5", lr="0.00001")
        with engine.session(self.app.config["DB_PATH"]) as conn:
            run_id = engine.list_runs(conn)[0]["id"]
        self.client.post(f"/runs/{run_id}/start", data={"pause_after": ""})
        for _ in range(500):
            with engine.session(self.app.config["DB_PATH"]) as conn:
                row = engine.get_run(conn, run_id)
            if row["updates_done"] >= 2:
                break
            time.sleep(0.002)
        self.client.post(f"/runs/{run_id}/pause")
        body = wait_idle(self.app, run_id)
        self.assertEqual(body["run"]["status"], "paused")
        self.assertEqual(body["run"]["pause_requested"], 0)
        self.assertLess(body["run"]["samples_done"], 120000)
        # resume to the end
        self.client.post(f"/runs/{run_id}/start", data={"pause_after": ""})
        body = wait_idle(self.app, run_id)
        self.assertEqual(body["run"]["status"], "finished")
        self.assertEqual(body["run"]["samples_done"], 120000)

    def test_bad_upload_is_rejected(self):
        resp = self.upload(csv=(io.BytesIO(b"x,y\n1,2\n3\n"), "bad.csv"))
        self.assertEqual(resp.status_code, 302)
        with engine.session(self.app.config["DB_PATH"]) as conn:
            self.assertEqual(engine.list_runs(conn), [])

    def test_param_validation_message(self):
        resp = self.upload(momentum="1.5")
        self.assertEqual(resp.status_code, 302)
        page = self.client.get("/").get_data(as_text=True)
        self.assertIn("动量系数", page)

    def test_unicode_filename_kept_and_stored_safely(self):
        resp = self.client.post("/upload", data={
            "csv": (io.BytesIO(b"x,y\n1,2\n2,4\n"), "实测校准.csv"),
            "group_size": "1", "chunk_limit": "1", "lr": "0.01",
            "momentum": "0.9", "init_w": "0", "init_b": "0",
        }, content_type="multipart/form-data", follow_redirects=False)
        self.assertEqual(resp.status_code, 302)
        with engine.session(self.app.config["DB_PATH"]) as conn:
            run = engine.list_runs(conn)[0]
        self.assertEqual(run["original_name"], "实测校准.csv")
        self.assertTrue(run["stored_name"].endswith(".csv"))
        self.assertIn("data", run["stored_name"])
        path = engine.resolve_data_path(self.app.config["UPLOAD_DIR"], run["stored_name"])
        self.assertTrue(os.path.exists(path))

    def test_wrong_extension_rejected(self):
        resp = self.client.post("/upload", data={
            "csv": (io.BytesIO(b"x,y\n1,2\n"), "payload.csv.exe"),
            "group_size": "1", "chunk_limit": "1", "lr": "0.01",
            "momentum": "0.9", "init_w": "0", "init_b": "0",
        }, content_type="multipart/form-data", follow_redirects=False)
        self.assertEqual(resp.status_code, 302)
        with engine.session(self.app.config["DB_PATH"]) as conn:
            self.assertEqual(engine.list_runs(conn), [])

    def test_delete_removes_file(self):
        self.upload()
        with engine.session(self.app.config["DB_PATH"]) as conn:
            run = engine.list_runs(conn)[0]
        path = engine.resolve_data_path(self.app.config["UPLOAD_DIR"], run["stored_name"])
        self.assertTrue(os.path.exists(path))
        self.client.post(f"/runs/{run['id']}/delete")
        self.assertFalse(os.path.exists(path))
        with engine.session(self.app.config["DB_PATH"]) as conn:
            self.assertEqual(engine.list_runs(conn), [])

    def test_recover_marks_interrupted_run(self):
        self.upload()
        with engine.session(self.app.config["DB_PATH"]) as conn:
            run_id = engine.list_runs(conn)[0]["id"]
            with engine.transaction(conn):
                conn.execute("UPDATE runs SET status='running' WHERE id=?", (run_id,))
        self.app2 = create_app({
            "DB_PATH": self.app.config["DB_PATH"],
            "UPLOAD_DIR": self.app.config["UPLOAD_DIR"],
            "SECRET_KEY": "test",
            "TESTING": True,
        })
        with engine.session(self.app.config["DB_PATH"]) as conn:
            self.assertEqual(engine.get_run(conn, run_id)["status"], "crashed")


if __name__ == "__main__":
    unittest.main(verbosity=2)
