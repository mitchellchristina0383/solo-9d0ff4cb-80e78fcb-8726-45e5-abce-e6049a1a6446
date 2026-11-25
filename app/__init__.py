"""Flask application factory for the streaming linear-regression trainer."""

from __future__ import annotations

import contextlib
import os
import threading
import time

from flask import (
    Flask,
    abort,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)
from werkzeug.utils import secure_filename

from trainer import engine

ALLOWED_EXT = {".csv", ".txt"}

_workers: dict[int, threading.Thread] = {}
_workers_lock = threading.Lock()


def worker_running(run_id: int) -> bool:
    with _workers_lock:
        thread = _workers.get(run_id)
        return thread is not None and thread.is_alive()


def _worker_main(db_path: str, upload_dir: str, run_id: int) -> None:
    conn = engine.connect(db_path)
    try:
        row = engine.get_run(conn, run_id)
        path = engine.resolve_data_path(upload_dir, row["stored_name"])
        engine.train(conn, run_id, path)
    except (engine.StateError, engine.InputError) as exc:
        engine._set_status(conn, run_id, "failed", str(exc))
    except BaseException as exc:  # noqa: BLE001 - keep the process and the DB consistent
        engine._set_status(
            conn, run_id, "crashed", f"训练线程异常: {type(exc).__name__}: {exc}"
        )
    finally:
        conn.close()
        with _workers_lock:
            if _workers.get(run_id) is threading.current_thread():
                _workers.pop(run_id, None)


def start_worker(app: Flask, run_id: int) -> bool:
    with _workers_lock:
        existing = _workers.get(run_id)
        if existing is not None and existing.is_alive():
            return False
        thread = threading.Thread(
            target=_worker_main,
            args=(app.config["DB_PATH"], app.config["UPLOAD_DIR"], run_id),
            name=f"train-{run_id}",
            daemon=True,
        )
        _workers[run_id] = thread
    thread.start()
    return True


def _int_field(name: str):
    raw = (request.form.get(name) or "").strip()
    if raw == "":
        return None
    try:
        return int(raw)
    except ValueError as exc:
        raise engine.InputError(f"字段 {name} 必须是整数") from exc


def _float_field(name: str):
    raw = (request.form.get(name) or "").strip()
    try:
        return float(raw)
    except ValueError as exc:
        raise engine.InputError(f"字段 {name} 必须是数值") from exc


def create_app(config: dict | None = None) -> Flask:
    app = Flask(__name__)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.environ.get("APP_DATA_DIR", os.path.join(base_dir, "..", "data"))
    app.config.update(
        DATA_DIR=data_dir,
        UPLOAD_DIR=os.environ.get("APP_UPLOAD_DIR", os.path.join(data_dir, "uploads")),
        DB_PATH=os.environ.get("APP_DB_PATH", os.path.join(data_dir, "trainer.sqlite3")),
        SECRET_KEY=os.environ.get("APP_SECRET_KEY", "local-dev-only"),
        MAX_CONTENT_LENGTH=int(os.environ.get("APP_MAX_UPLOAD_MB", "256")) * 1024 * 1024,
    )
    if config:
        app.config.update(config)
    os.makedirs(app.config["UPLOAD_DIR"], exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(app.config["DB_PATH"])) or ".", exist_ok=True)

    with engine.session(app.config["DB_PATH"]) as conn:
        engine.recover(conn)

    def db():
        return engine.session(app.config["DB_PATH"])

    @app.get("/")
    def index():
        with db() as conn:
            runs = engine.list_runs(conn)
        return render_template(
            "index.html",
            runs=runs,
            max_upload_mb=app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024),
        )

    @app.post("/upload")
    def upload():
        file = request.files.get("csv")
        if file is None or not file.filename:
            flash("请选择 CSV 文件", "error")
            return redirect(url_for("index"))
        # Keep the user-visible name (Chinese names included); sanitise only the
        # on-disk name.  Browsers may send a full path in old versions.
        original = os.path.basename(file.filename.replace("\\", "/")).strip()[:200]
        ext = os.path.splitext(original)[1].lower()
        if ext not in ALLOWED_EXT:
            flash("只支持 .csv / .txt 文件", "error")
            return redirect(url_for("index"))
        # secure_filename drops non-ASCII characters entirely (e.g. 实测.csv ->
        # "csv"), so keep the extension attached to the sanitised stem.
        stem = secure_filename(os.path.splitext(original)[0]) or "data"
        safe = f"{stem}{ext}"

        try:
            group_size = _int_field("group_size")
            chunk_limit = _int_field("chunk_limit")
            if group_size is None:
                raise engine.InputError("请填写每组样本数")
            if chunk_limit is None:
                raise engine.InputError("请填写单次读入行数上限")
            lr = _float_field("lr")
            momentum = _float_field("momentum")
            init_w = _float_field("init_w")
            init_b = _float_field("init_b")
            engine.validate_params(
                group_size, chunk_limit, lr, momentum, init_w, init_b, None
            )
        except engine.InputError as exc:
            flash(str(exc), "error")
            return redirect(url_for("index"))

        stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        stored = f"{stamp}-{os.getpid()}-{time.monotonic_ns()}-{safe}"
        path = os.path.join(app.config["UPLOAD_DIR"], stored)
        file.save(path)

        run_id = None
        try:
            with db() as conn:
                run_id = engine.create_run(
                    conn,
                    path,
                    original,
                    stored,
                    group_size=group_size,
                    chunk_limit=chunk_limit,
                    lr=lr,
                    momentum=momentum,
                    init_w=init_w,
                    init_b=init_b,
                )
        except engine.InputError as exc:
            with contextlib.suppress(OSError):
                os.unlink(path)
            flash(str(exc), "error")
            return redirect(url_for("index"))

        flash("数据已上传并校验通过。设置暂停点后开始训练。", "ok")
        return redirect(url_for("run_detail", run_id=run_id))

    @app.get("/runs/<int:run_id>")
    def run_detail(run_id: int):
        with db() as conn:
            try:
                run = engine.get_run(conn, run_id)
            except engine.InputError:
                abort(404)
            updates = engine.recent_updates(conn, run_id, 15)
        return render_template(
            "run.html",
            run=run,
            updates=updates,
            worker_running=worker_running(run_id),
        )

    @app.get("/api/runs/<int:run_id>")
    def api_run(run_id: int):
        with db() as conn:
            try:
                run = engine.get_run(conn, run_id)
            except engine.InputError:
                abort(404)
            updates = engine.recent_updates(conn, run_id, 15)
        return jsonify(
            run=dict(run),
            worker_running=worker_running(run_id),
            updates=[dict(u) for u in updates],
        )

    @app.get("/api/runs/<int:run_id>/audit")
    def api_audit(run_id: int):
        with db() as conn:
            try:
                result = engine.audit(conn, run_id)
            except engine.InputError:
                abort(404)
        result["run"] = dict(result["run"])
        return jsonify(result)

    @app.post("/runs/<int:run_id>/start")
    def run_start(run_id: int):
        pause_after = None
        pause_desc = "，将跑到结束"
        with db() as conn:
            try:
                run = engine.get_run(conn, run_id)
                if run["status"] == "finished":
                    raise engine.InputError("该训练已完成")
                if worker_running(run_id):
                    raise engine.InputError("训练正在进行中")
                chunk_limit = _int_field("chunk_limit")
                if chunk_limit is not None and chunk_limit != run["chunk_limit"]:
                    engine.set_chunk_limit(conn, run_id, chunk_limit)
                pause_after = _int_field("pause_after")
                pause_unit = (request.form.get("pause_unit") or "chunk").strip()
                if pause_after is not None and pause_after < 1:
                    raise engine.InputError("暂停数量必须 ≥ 1，或留空表示跑完")
                if pause_unit not in ("chunk", "group"):
                    pause_unit = "chunk"
                with engine.transaction(conn):
                    conn.execute(
                        "UPDATE runs SET pause_after_chunks=CASE WHEN ?='chunk' THEN ? END, "
                        "pause_after_groups=CASE WHEN ?='group' THEN ? END, "
                        "pause_requested=0, message=?, updated_at=? WHERE id=?",
                        (pause_unit, pause_after, pause_unit, pause_after,
                         ("已排队，等待训练线程" if not pause_after
                          else f"已排队，将在 {pause_after} "
                               + ("段后暂停" if pause_unit == "chunk" else "组更新后暂停")),
                         engine._now(), run_id),
                    )
                pause_desc = (
                    f"，将在读完 {pause_after} 段后暂停" if pause_after and pause_unit == "chunk"
                    else f"，将在完成 {pause_after} 组更新后暂停" if pause_after
                    else "，将跑到结束"
                )
            except (engine.InputError, engine.StateError) as exc:
                flash(str(exc), "error")
                return redirect(url_for("run_detail", run_id=run_id))

        if not start_worker(app, run_id):
            flash("后台训练线程仍在收尾，请稍后重试", "error")
        else:
            flash("训练已启动" + pause_desc, "ok")
        return redirect(url_for("run_detail", run_id=run_id))

    @app.post("/runs/<int:run_id>/pause")
    def run_pause(run_id: int):
        with db() as conn:
            try:
                ok = engine.request_pause(conn, run_id)
            except engine.InputError:
                abort(404)
        flash(
            "已提交暂停请求：当前读取段结束即落盘暂停，下次从下一行继续"
            if ok else "当前没有运行中的训练",
            "ok" if ok else "error",
        )
        return redirect(url_for("run_detail", run_id=run_id))

    @app.post("/runs/<int:run_id>/chunk-limit")
    def run_chunk_limit(run_id: int):
        with db() as conn:
            try:
                value = _int_field("chunk_limit")
                if value is None:
                    raise engine.InputError("请填写读取上限")
                engine.set_chunk_limit(conn, run_id, value)
                flash(f"读取上限已设置为 {value} 行，将在下次读取时生效", "ok")
            except (engine.InputError, engine.StateError) as exc:
                flash(str(exc), "error")
        return redirect(url_for("run_detail", run_id=run_id))

    @app.post("/runs/<int:run_id>/delete")
    def run_delete(run_id: int):
        with db() as conn:
            try:
                row = engine.delete_run(conn, run_id)
            except (engine.InputError, engine.StateError) as exc:
                flash(str(exc), "error")
                return redirect(url_for("run_detail", run_id=run_id))
        with contextlib.suppress(OSError):
            os.unlink(engine.resolve_data_path(app.config["UPLOAD_DIR"], row["stored_name"]))
        flash(f"已删除训练 #{run_id} 的记录与数据文件", "ok")
        return redirect(url_for("index"))

    @app.get("/uploads/<path:name>")
    def download_upload(name: str):
        return send_from_directory(app.config["UPLOAD_DIR"], name, as_attachment=True)

    @app.errorhandler(413)
    def too_large(_exc):
        flash("文件超过上传大小限制", "error")
        return redirect(url_for("index"))

    return app


__all__ = ["create_app", "start_worker", "worker_running"]
