"""Crash-consistent, memory-bounded univariate linear regression trainer.

Model: ``y_hat = w * x + b`` with squared-error loss.
Optimiser: one heavy-ball momentum update per consecutive group of rows::

    v <- momentum * v - lr * grad
    w <- w + v

The CSV has two columns (x = measured value, y = reference value).  A leading
header row is accepted and ignored; rows are consumed in file order.  A group is
a consecutive run of ``group_size`` data rows (blank lines are skipped).  The
final group may be shorter and is still updated exactly once.

Memory bound
------------
Rows are read ``chunk_limit`` at a time into a temporary list, the gradient and
loss of that chunk are accumulated row by row, and the list is dropped.  Peak
residency is therefore ``min(chunk_limit, group_size)`` samples, independent of
file length; the regression state itself is a handful of doubles.

Durability
----------
SQLite (WAL, ``synchronous=FULL``) keeps the parameters, the momentum, the
sample/update progress and the in-group accumulator in one row.  Progress
advances in two kinds of commit:

* **group commit** (once per update): parameters, momentum, ``samples_done``,
  ``updates_done`` and an audit row are written in one transaction.  The group
  update is therefore all-or-nothing.
* **pause commit** (only when deliberately pausing, at a chunk boundary): the
  partial in-group accumulator and the file offset are written.  A deliberate
  pause resumes at the *next row* -- no row is read twice.

An unexpected exit never gets to run the pause commit, so the run restarts from
the last group commit and only the group that had not completed re-reads; every
already-counted group is never re-read.

Floating-point exactness: rows are accumulated one at a time in file order into
the running sums, and SQLite stores IEEE-754 doubles.  Chunk size or a pause
therefore cannot change any rounding, and a resumed run reproduces the
uninterrupted run bit-for-bit on the same platform.
"""

from __future__ import annotations

import contextlib
import csv
import hashlib
import math
import os
import sqlite3
import time
from dataclasses import dataclass
from typing import Callable, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    original_name      TEXT NOT NULL,
    stored_name        TEXT NOT NULL,
    file_size          INTEGER NOT NULL,
    file_sha256        TEXT NOT NULL,
    data_start         INTEGER NOT NULL,
    has_header         INTEGER NOT NULL,
    total_rows         INTEGER NOT NULL,
    group_size         INTEGER NOT NULL,
    chunk_limit        INTEGER NOT NULL,
    lr                 REAL NOT NULL,
    momentum           REAL NOT NULL,
    init_w             REAL NOT NULL,
    init_b             REAL NOT NULL,
    w                  REAL NOT NULL,
    b                  REAL NOT NULL,
    v_w                REAL NOT NULL,
    v_b                REAL NOT NULL,
    next_offset        INTEGER NOT NULL,
    samples_done       INTEGER NOT NULL,
    updates_done       INTEGER NOT NULL,
    part_rows          INTEGER NOT NULL DEFAULT 0,
    part_gw            REAL NOT NULL DEFAULT 0,
    part_gb            REAL NOT NULL DEFAULT 0,
    part_loss          REAL NOT NULL DEFAULT 0,
    part_start_offset  INTEGER NOT NULL DEFAULT 0,
    status             TEXT NOT NULL,          -- ready|running|paused|finished|crashed|failed
    message            TEXT NOT NULL,
    pause_after_chunks INTEGER,                -- pause once this many chunks are read (per session)
    pause_after_groups INTEGER,                -- or pause after this many group updates (per session)
    pause_requested    INTEGER NOT NULL DEFAULT 0,
    chunks_read        INTEGER NOT NULL DEFAULT 0,
    peak_rows_held     INTEGER NOT NULL DEFAULT 0,
    peak_rss_kb        INTEGER NOT NULL DEFAULT 0,
    last_group_ms      REAL NOT NULL DEFAULT 0,
    created_at         TEXT NOT NULL,
    updated_at         TEXT NOT NULL,
    finished_at        TEXT
);

CREATE TABLE IF NOT EXISTS updates (
    run_id        INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    idx           INTEGER NOT NULL,
    samples       INTEGER NOT NULL,
    offset_before INTEGER NOT NULL,
    offset_after  INTEGER NOT NULL,
    w_before      REAL NOT NULL,
    b_before      REAL NOT NULL,
    grad_w        REAL NOT NULL,
    grad_b        REAL NOT NULL,
    loss_mean     REAL NOT NULL,
    v_w           REAL NOT NULL,
    v_b           REAL NOT NULL,
    w_after       REAL NOT NULL,
    b_after       REAL NOT NULL,
    group_ms      REAL NOT NULL,
    created_at    TEXT NOT NULL,
    PRIMARY KEY (run_id, idx)
);

CREATE UNIQUE INDEX IF NOT EXISTS ux_runs_open
    ON runs(original_name)
    WHERE status IN ('ready','running','paused','crashed');
"""

# Columns introduced over time; ensure older databases keep working.
_ADDED_COLUMNS = {
    "part_rows": "INTEGER NOT NULL DEFAULT 0",
    "part_gw": "REAL NOT NULL DEFAULT 0",
    "part_gb": "REAL NOT NULL DEFAULT 0",
    "part_loss": "REAL NOT NULL DEFAULT 0",
    "part_start_offset": "INTEGER NOT NULL DEFAULT 0",
    "pause_after_chunks": "INTEGER",
    "pause_after_groups": "INTEGER",
    "chunks_read": "INTEGER NOT NULL DEFAULT 0",
    "peak_rows_held": "INTEGER NOT NULL DEFAULT 0",
    "peak_rss_kb": "INTEGER NOT NULL DEFAULT 0",
    "last_group_ms": "REAL NOT NULL DEFAULT 0",
}


class InputError(ValueError):
    """Bad request / bad data: shown to the user, run becomes 'failed'."""


class StateError(RuntimeError):
    """Requested transition is not allowed in the current state."""


# --------------------------------------------------------------------------- #
# CSV handling
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FileMeta:
    data_start: int
    rows: int
    has_header: bool


def decode_line(raw: bytes) -> str:
    """Decode one line; plain ASCII (the normal case) skips the BOM codec."""
    if raw.isascii():
        return raw.decode("ascii")
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


MAX_LINE_BYTES = 1 << 20  # a row of two numbers is tiny; this only bounds pathology


def read_line(fh) -> bytes:
    """Read one line with a hard cap so a malformed file cannot exhaust memory."""
    raw = fh.readline(MAX_LINE_BYTES)
    if len(raw) >= MAX_LINE_BYTES and not raw.endswith(b"\n"):
        raise InputError(f"文件中有超过 {MAX_LINE_BYTES} 字节的超长行")
    return raw


def parse_pair(text: str, lineno: Optional[int] = None) -> tuple[float, float]:
    """Parse one physical line into (x, y).

    Fast path for plain ``a,b`` lines; anything else goes through the csv module
    so quoted fields still work.
    """
    if "," in text and '"' not in text:
        left, right = text.split(",", 1)
        try:
            x = float(left)
            y = float(right)
        except ValueError:
            pass
        else:
            if math.isfinite(x) and math.isfinite(y):
                return x, y
            where = f"第 {lineno} 行" if lineno else "数据行"
            raise InputError(f"{where}含 NaN/Inf")
    row = next(csv.reader([text]))
    if len(row) != 2:
        where = f"第 {lineno} 行" if lineno else "数据行"
        raise InputError(f"{where}应有 2 列（实测值、参考值），实际 {len(row)} 列")
    try:
        x = float(row[0])
        y = float(row[1])
    except ValueError as exc:
        where = f"第 {lineno} 行" if lineno else "数据行"
        raise InputError(f"{where}不是数值: {text.strip()[:60]!r}") from exc
    if not (math.isfinite(x) and math.isfinite(y)):
        where = f"第 {lineno} 行" if lineno else "数据行"
        raise InputError(f"{where}含 NaN/Inf")
    return x, y


def scan_file(path: str) -> FileMeta:
    """Validate the whole file and count data rows without holding it in memory."""
    try:
        size = os.path.getsize(path)
    except OSError as exc:
        raise InputError(f"无法读取文件: {exc}") from exc
    if size == 0:
        raise InputError("文件为空")
    rows = 0
    data_start: Optional[int] = None
    has_header = False
    first = True
    with open(path, "rb") as fh:
        lineno = 0
        while True:
            start = fh.tell()
            raw = read_line(fh)
            if not raw:
                break
            lineno += 1
            text = decode_line(raw).strip()
            if not text:
                continue
            if first:
                first = False
                cols = next(csv.reader([text]))
                if len(cols) < 2:
                    raise InputError(
                        f"第 {lineno} 行需要至少两列（实测值、参考值），实际 {len(cols)} 列"
                    )
                try:
                    float(cols[0])
                    float(cols[1])
                except ValueError:
                    has_header = True
                    data_start = fh.tell()
                    continue
                data_start = start
            elif data_start is None:
                data_start = start
            parse_pair(text, lineno)
            rows += 1
    if data_start is None:
        raise InputError("文件中没有可读内容")
    if rows < 1:
        raise InputError("只有表头，没有数据行")
    return FileMeta(data_start=data_start, rows=rows, has_header=has_header)


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def _finite(name: str, value: float) -> float:
    v = float(value)
    if not math.isfinite(v):
        raise InputError(f"{name} 必须是有限数值")
    return v


def validate_params(
    group_size: int,
    chunk_limit: int,
    lr: float,
    momentum: float,
    init_w: float,
    init_b: float,
    pause_after: Optional[int],
) -> dict:
    if not isinstance(group_size, int) or group_size < 1:
        raise InputError("每组样本数必须是 ≥ 1 的整数")
    if not isinstance(chunk_limit, int) or chunk_limit < 1:
        raise InputError("单次读入行数上限必须是 ≥ 1 的整数")
    lr = _finite("学习率", lr)
    momentum = _finite("动量系数", momentum)
    init_w = _finite("初始 w", init_w)
    init_b = _finite("初始 b", init_b)
    if lr <= 0:
        raise InputError("学习率必须大于 0")
    if not (0.0 <= momentum < 1.0):
        raise InputError("动量系数必须在 [0, 1) 区间")
    if pause_after is not None:
        if not isinstance(pause_after, int) or pause_after < 1:
            raise InputError("“读完几段后暂停”必须是 ≥ 1 的整数或留空")
    return dict(
        group_size=group_size,
        chunk_limit=chunk_limit,
        lr=lr,
        momentum=momentum,
        init_w=init_w,
        init_b=init_b,
        pause_after=pause_after,
    )


# --------------------------------------------------------------------------- #
# Database
# --------------------------------------------------------------------------- #
def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def connect(path: str) -> sqlite3.Connection:
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=FULL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA cache_size=-2000")  # ~2 MB page cache: small machines
    conn.executescript(SCHEMA)
    _migrate(conn)
    return conn


@contextlib.contextmanager
def session(path: str):
    """Open a connection and always close it (``with sqlite3.connect()`` does not)."""
    conn = connect(path)
    try:
        yield conn
    finally:
        conn.close()


def _migrate(conn: sqlite3.Connection) -> None:
    have = {row["name"] for row in conn.execute("PRAGMA table_info(runs)")}
    for column, decl in _ADDED_COLUMNS.items():
        if column not in have:
            conn.execute(f"ALTER TABLE runs ADD COLUMN {column} {decl}")


@contextlib.contextmanager
def transaction(conn: sqlite3.Connection):
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
        conn.execute("COMMIT")
    except BaseException:
        with contextlib.suppress(sqlite3.Error):
            conn.execute("ROLLBACK")
        raise


def recover(conn: sqlite3.Connection) -> int:
    """Startup recovery: a 'running' row can only be left by a dead process.

    The row already points at the last group commit (or last deliberate pause),
    so nothing else has to be repaired; the flag just tells the UI why it is
    not running any more.
    """
    with transaction(conn):
        cur = conn.execute(
            "UPDATE runs SET status='crashed', updated_at=?, "
            "message='上次进程在训练中被终止；已回退到最后一个已提交的组边界，可继续训练' "
            "WHERE status='running'",
            (_now(),),
        )
        return cur.rowcount


def resolve_data_path(upload_dir: str, stored_name: str) -> str:
    if os.path.isabs(stored_name):
        return stored_name
    return os.path.join(upload_dir, stored_name)


def create_run(
    conn: sqlite3.Connection,
    data_path: str,
    original_name: str,
    stored_name: str,
    *,
    group_size: int,
    chunk_limit: int,
    lr: float,
    momentum: float,
    init_w: float,
    init_b: float,
    pause_after: Optional[int] = None,
) -> int:
    params = validate_params(group_size, chunk_limit, lr, momentum, init_w, init_b, pause_after)
    meta = scan_file(data_path)
    size = os.path.getsize(data_path)
    digest = sha256_file(data_path)
    now = _now()
    with transaction(conn):
        dup = conn.execute(
            "SELECT id FROM runs WHERE original_name=? AND status IN "
            "('ready','running','paused','crashed')",
            (original_name,),
        ).fetchone()
        if dup:
            raise InputError(f"同名文件已有未结束的训练（#{dup['id']}），请先完成或删除")
        cur = conn.execute(
            """INSERT INTO runs(original_name, stored_name, file_size, file_sha256,
                                data_start, has_header, total_rows, group_size, chunk_limit,
                                lr, momentum, init_w, init_b, w, b, v_w, v_b,
                                next_offset, samples_done, updates_done,
                                part_start_offset, status, message, pause_after_chunks,
                                created_at, updated_at)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                original_name, stored_name, size, digest,
                meta.data_start, 1 if meta.has_header else 0, meta.rows,
                params["group_size"], params["chunk_limit"], params["lr"],
                params["momentum"], params["init_w"], params["init_b"],
                params["init_w"], params["init_b"], 0.0, 0.0,
                meta.data_start, 0, 0,
                meta.data_start, "ready", "已创建，尚未开始",
                params["pause_after"], now, now,
            ),
        )
        return int(cur.lastrowid)


def get_run(conn: sqlite3.Connection, run_id: int) -> sqlite3.Row:
    row = conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
    if row is None:
        raise InputError(f"训练 #{run_id} 不存在")
    return row


def list_runs(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute("SELECT * FROM runs ORDER BY id DESC").fetchall()


def recent_updates(conn: sqlite3.Connection, run_id: int, limit: int = 20) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM updates WHERE run_id=? ORDER BY idx DESC LIMIT ?",
        (run_id, max(1, int(limit))),
    ).fetchall()


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
@dataclass
class Part:
    """In-group accumulator: gradient/loss sums of rows already read."""

    rows: int = 0
    gw: float = 0.0
    gb: float = 0.0
    loss: float = 0.0
    start_offset: int = 0

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Part":
        return cls(int(row["part_rows"]), float(row["part_gw"]),
                   float(row["part_gb"]), float(row["part_loss"]),
                   int(row["part_start_offset"]) or int(row["next_offset"]))

    def reset(self) -> None:
        self.rows = 0
        self.gw = 0.0
        self.gb = 0.0
        self.loss = 0.0
        self.start_offset = 0


@dataclass
class ChunkResult:
    rows: int
    offset_after: int


def read_chunk(fh, offset: int, want: int, w: float, b: float, part: Part,
               chunk_limit: int) -> ChunkResult:
    """Read up to ``want`` rows into a bounded list, then fold it into ``part``.

    The list holds at most ``chunk_limit`` samples and is dropped before the
    caller continues, so resident samples never exceed the configured limit.
    """
    fh.seek(offset)
    batch: list[tuple[float, float]] = []
    readline = fh.readline
    append = batch.append
    # local copies of the running sums: same per-row addition order as writing
    # straight into ``part``, so chunk size cannot change any rounding
    gw, gb, loss = part.gw, part.gb, part.loss
    while len(batch) < want:
        raw = readline(MAX_LINE_BYTES)
        if not raw:
            break
        if len(raw) >= MAX_LINE_BYTES and raw[-1:] != b"\n":
            raise InputError(f"文件中有超过 {MAX_LINE_BYTES} 字节的超长行")
        text = decode_line(raw)
        if not text.strip():
            continue
        append(parse_pair(text))
    offset_after = fh.tell()
    for x, y in batch:
        err = w * x + b - y
        gw += err * x
        gb += err
        loss += err * err
    part.gw = gw
    part.gb = gb
    part.loss = loss
    rows_in_chunk = len(batch)
    part.rows += rows_in_chunk
    del batch  # release the chunk before the next read
    return ChunkResult(rows_in_chunk, offset_after)


@dataclass
class TrainResult:
    status: str
    message: str
    updates: int
    samples: int
    total_rows: int
    w: float
    b: float
    v_w: float
    v_b: float
    part_rows: int = 0
    chunks_this_session: int = 0


def _peak_rss_kb() -> int:
    try:
        with open("/proc/self/status", "r", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    return 0


def _set_status(conn: sqlite3.Connection, run_id: int, status: str, message: str) -> None:
    try:
        with transaction(conn):
            conn.execute(
                "UPDATE runs SET status=?, message=?, updated_at=?, "
                "finished_at=CASE WHEN ?='finished' THEN ? ELSE finished_at END "
                "WHERE id=?",
                (status, message, _now(), status, _now(), run_id),
            )
    except sqlite3.Error:
        pass


def _verify_file(run: sqlite3.Row, data_path: str) -> None:
    try:
        size = os.path.getsize(data_path)
    except OSError as exc:
        raise InputError(f"数据文件不可用: {exc}") from exc
    if size != run["file_size"] or sha256_file(data_path) != run["file_sha256"]:
        raise InputError("数据文件已变化（大小或 SHA-256 不符），拒绝继续训练")


def train(
    conn: sqlite3.Connection,
    run_id: int,
    data_path: str,
    *,
    max_updates: Optional[int] = None,
    crash_hook: Optional[Callable[[str, dict], None]] = None,
    progress: Optional[Callable[[dict], None]] = None,
) -> TrainResult:
    """Run or resume.

    ``max_updates`` caps the number of group updates committed in this call
    (tests/CLI).  The web control is ``pause_after_chunks``: pause once that
    many read chunks have been consumed in this session, checkpointing the
    partial group so the next session starts at the following row.
    ``crash_hook`` is a test-only injection point; raising from it simulates an
    abrupt exit.
    """
    run = get_run(conn, run_id)
    if run["status"] == "finished":
        return TrainResult("finished", run["message"], run["updates_done"],
                           run["samples_done"], run["total_rows"],
                           run["w"], run["b"], run["v_w"], run["v_b"],
                           int(run["part_rows"]))
    if run["status"] == "failed":
        raise StateError(f"训练处于失败状态，不能继续：{run['message']}")

    _verify_file(run, data_path)

    total = int(run["total_rows"])
    group_size = int(run["group_size"])
    chunk_limit = int(run["chunk_limit"])
    lr = float(run["lr"])
    momentum = float(run["momentum"])
    w = float(run["w"])
    b = float(run["b"])
    v_w = float(run["v_w"])
    v_b = float(run["v_b"])
    offset = int(run["next_offset"])
    samples_done = int(run["samples_done"])
    updates_done = int(run["updates_done"])
    peak_rows_held = int(run["peak_rows_held"])
    peak_rss = int(run["peak_rss_kb"])
    chunks_total = int(run["chunks_read"])
    part = Part.from_row(run)
    chunk_budget = run["pause_after_chunks"]
    chunk_budget = int(chunk_budget) if chunk_budget is not None else None
    group_budget = run["pause_after_groups"]
    group_budget = int(group_budget) if group_budget is not None else None
    chunks_session = 0
    groups_session = 0
    status = "paused"
    message = "训练未产生新的更新"

    with transaction(conn):
        cur = conn.execute(
            "UPDATE runs SET status='running', pause_requested=0, message='训练中', "
            "updated_at=? WHERE id=? AND status IN ('ready','paused','crashed')",
            (_now(), run_id),
        )
        if cur.rowcount == 0:
            raise StateError("当前状态不能开始训练（可能已有训练线程在运行）")
    if progress:
        progress({"event": "status", "status": "running", "message": "训练中"})

    def pause_check() -> Optional[str]:
        row = conn.execute(
            "SELECT pause_requested FROM runs WHERE id=?", (run_id,)
        ).fetchone()
        if row["pause_requested"]:
            return "收到暂停请求：已在当前读取段结束后暂停，下次从下一行继续"
        if max_updates is not None and groups_session >= max_updates:
            return f"本次运行达到更新上限（{max_updates} 组）：已在组边界暂停"
        if group_budget is not None and groups_session >= group_budget:
            return f"本次已完成 {group_budget} 组更新：按设置暂停"
        if chunk_budget is not None and chunks_session >= chunk_budget:
            return f"本次已读完 {chunk_budget} 段：按设置暂停，下次从下一行继续"
        return None

    try:
        with open(data_path, "rb") as fh:
            while samples_done + part.rows < total:
                reason = pause_check()
                if reason:
                    status, message = "paused", reason
                    break
                group_total = min(group_size, total - samples_done)
                want = min(group_total - part.rows, chunk_limit)
                if part.rows == 0:
                    part.start_offset = offset
                if progress:
                    progress({
                        "event": "chunk_start",
                        "group_index": updates_done,
                        "part_rows": part.rows,
                        "group_rows": group_total,
                        "offset": offset,
                    })
                t0 = time.perf_counter()
                chunk = read_chunk(fh, offset, want, w, b, part, chunk_limit)
                if chunk.rows == 0:
                    raise InputError("读取数据提前结束，文件可能被截断")
                offset = chunk.offset_after
                chunks_session += 1
                chunks_total += 1
                peak_rows_held = max(peak_rows_held, chunk.rows)
                if progress:
                    progress({
                        "event": "chunk",
                        "group_index": updates_done,
                        "rows_in_chunk": chunk.rows,
                        "rows_this_group": part.rows,
                        "group_rows": group_total,
                        "offset_after": offset,
                        "samples_done": samples_done,
                        "chunks_session": chunks_session,
                    })
                if crash_hook:
                    crash_hook("chunk", {
                        "epoch": updates_done,
                        "rows_in_chunk": chunk.rows,
                        "rows_this_group": part.rows,
                        "group_rows": group_total,
                        "samples_done": samples_done,
                        "chunk_index": chunks_session - 1,
                    })

                if part.rows < group_total:
                    continue  # more chunks needed for this group

                # ---------- complete group: one momentum update ----------
                grad_w = 2.0 * part.gw / part.rows
                grad_b = 2.0 * part.gb / part.rows
                loss_mean = part.loss / part.rows
                new_v_w = momentum * v_w - lr * grad_w
                new_v_b = momentum * v_b - lr * grad_b
                new_w = w + new_v_w
                new_b = b + new_v_b
                if not (math.isfinite(new_w) and math.isfinite(new_b)
                        and math.isfinite(new_v_w) and math.isfinite(new_v_b)):
                    raise InputError(
                        "训练发散：参数或动量出现 NaN/Inf。请减小学习率、"
                        "放大每组样本数，或检查数据量级"
                    )
                new_samples = samples_done + part.rows
                new_updates = updates_done + 1
                group_ms = (time.perf_counter() - t0) * 1000.0
                rss = _peak_rss_kb()
                samples_this_group = part.rows
                if crash_hook:
                    crash_hook("before_commit", {
                        "epoch": updates_done, "samples_done": samples_done,
                        "new_samples_done": new_samples,
                        "w": new_w, "b": new_b,
                    })

                # Atomic: params + momentum + counted progress + audit trail.
                with transaction(conn):
                    conn.execute(
                        """INSERT INTO updates(run_id, idx, samples, offset_before, offset_after,
                                               w_before, b_before, grad_w, grad_b, loss_mean,
                                               v_w, v_b, w_after, b_after, group_ms, created_at)
                           VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (run_id, updates_done, samples_this_group,
                         part.start_offset, offset,
                         w, b, grad_w, grad_b, loss_mean,
                         new_v_w, new_v_b, new_w, new_b, group_ms, _now()),
                    )
                    conn.execute(
                        """UPDATE runs SET w=?, b=?, v_w=?, v_b=?, next_offset=?,
                                           samples_done=?, updates_done=?,
                                           part_rows=0, part_gw=0, part_gb=0, part_loss=0,
                                           part_start_offset=0,
                                           chunks_read=?, peak_rows_held=MAX(peak_rows_held, ?),
                                           peak_rss_kb=MAX(peak_rss_kb, ?),
                                           last_group_ms=?, updated_at=?, status='running',
                                           message='训练中'
                           WHERE id=?""",
                        (new_w, new_b, new_v_w, new_v_b, offset,
                         new_samples, new_updates,
                         chunks_total, peak_rows_held, rss,
                         group_ms, _now(), run_id),
                    )

                w, b, v_w, v_b = new_w, new_b, new_v_w, new_v_b
                samples_done, updates_done = new_samples, new_updates
                part.reset()
                groups_session += 1
                if progress:
                    progress({
                        "event": "committed", "epoch": new_updates - 1,
                        "updates_done": updates_done, "samples_done": samples_done,
                        "total_rows": total, "w": w, "b": b, "v_w": v_w, "v_b": v_b,
                        "group_ms": group_ms, "peak_rows_held": peak_rows_held,
                        "loss_mean": loss_mean, "chunks_session": chunks_session,
                    })
                if crash_hook:
                    crash_hook("after_commit", {
                        "epoch": new_updates - 1, "updates_done": updates_done,
                        "samples_done": samples_done,
                    })
            else:
                status, message = "finished", "训练完成"

        # Persist final state.  For a pause this also checkpoints the partial
        # group so the next session resumes at `offset` (the next row).
        with transaction(conn):
            conn.execute(
                "UPDATE runs SET status=?, message=?, pause_requested=0, "
                "next_offset=?, part_rows=?, part_gw=?, part_gb=?, part_loss=?, "
                "part_start_offset=?, chunks_read=?, "
                "peak_rows_held=MAX(peak_rows_held, ?), peak_rss_kb=MAX(peak_rss_kb, ?), "
                "updated_at=?, finished_at=CASE WHEN ?='finished' THEN ? ELSE finished_at END "
                "WHERE id=?",
                (status, message, offset, part.rows, part.gw, part.gb, part.loss,
                 part.start_offset, chunks_total, peak_rows_held, _peak_rss_kb(),
                 _now(), status, _now(), run_id),
            )
    except KeyboardInterrupt:
        _set_status(conn, run_id, "paused", "收到中断信号，已在当前读取段结束后暂停")
        if progress:
            progress({"event": "status", "status": "paused", "message": "收到中断信号"})
        raise
    except InputError as exc:
        _set_status(conn, run_id, "failed", str(exc))
        if progress:
            progress({"event": "status", "status": "failed", "message": str(exc)})
        raise
    except StateError:
        raise
    except BaseException as exc:  # noqa: BLE001 - record any other failure
        _set_status(conn, run_id, "crashed", f"训练异常退出: {type(exc).__name__}: {exc}")
        if progress:
            progress({"event": "status", "status": "crashed",
                      "message": f"{type(exc).__name__}: {exc}"})
        raise

    if progress:
        progress({"event": "status", "status": status, "message": message})
    return TrainResult(status, message, updates_done, samples_done, total,
                       w, b, v_w, v_b, part.rows, chunks_session)


def set_chunk_limit(conn: sqlite3.Connection, run_id: int, chunk_limit: int) -> None:
    if not isinstance(chunk_limit, int) or chunk_limit < 1:
        raise InputError("单次读入行数上限必须是 ≥ 1 的整数")
    with transaction(conn):
        row = get_run(conn, run_id)
        if row["status"] == "running":
            raise StateError("训练运行中，暂停后才能调整读取上限")
        if row["status"] == "finished":
            raise StateError("训练已结束，调整读取上限没有意义")
        conn.execute(
            "UPDATE runs SET chunk_limit=?, message=?, updated_at=? WHERE id=?",
            (chunk_limit, f"单次读入上限已调整为 {chunk_limit} 行", _now(), run_id),
        )


def set_pause_after_chunks(conn: sqlite3.Connection, run_id: int,
                           chunks: Optional[int]) -> None:
    if chunks is not None and (not isinstance(chunks, int) or chunks < 1):
        raise InputError("“读完几段后暂停”必须是 ≥ 1 的整数或留空")
    with transaction(conn):
        get_run(conn, run_id)
        conn.execute(
            "UPDATE runs SET pause_after_chunks=?, pause_after_groups=NULL, "
            "updated_at=? WHERE id=?",
            (chunks, _now(), run_id),
        )


def set_pause_after_groups(conn: sqlite3.Connection, run_id: int,
                           groups: Optional[int]) -> None:
    if groups is not None and (not isinstance(groups, int) or groups < 1):
        raise InputError("“完成几组后暂停”必须是 ≥ 1 的整数或留空")
    with transaction(conn):
        get_run(conn, run_id)
        conn.execute(
            "UPDATE runs SET pause_after_groups=?, pause_after_chunks=NULL, "
            "updated_at=? WHERE id=?",
            (groups, _now(), run_id),
        )


def request_pause(conn: sqlite3.Connection, run_id: int) -> bool:
    with transaction(conn):
        cur = conn.execute(
            "UPDATE runs SET pause_requested=1, "
            "message='暂停请求已提交，将在当前读取段结束时暂停并落盘', updated_at=? "
            "WHERE id=? AND status='running'",
            (_now(), run_id),
        )
        return cur.rowcount > 0


def delete_run(conn: sqlite3.Connection, run_id: int) -> sqlite3.Row:
    with transaction(conn):
        row = get_run(conn, run_id)
        if row["status"] == "running":
            raise StateError("训练运行中，不能删除")
        conn.execute("DELETE FROM runs WHERE id=?", (run_id,))
        return row


# --------------------------------------------------------------------------- #
# Verification helpers
# --------------------------------------------------------------------------- #
def reference_fit(
    data_path: str,
    *,
    data_start: int,
    total_rows: int,
    group_size: int,
    chunk_limit: int,
    lr: float,
    momentum: float,
    init_w: float,
    init_b: float,
    on_chunk: Optional[Callable[[dict], None]] = None,
) -> dict:
    """Uninterrupted, DB-free fit used to check resumed runs."""
    w, b, v_w, v_b = init_w, init_b, 0.0, 0.0
    done = 0
    offset = data_start
    updates = 0
    with open(data_path, "rb") as fh:
        while done < total_rows:
            group_total = min(group_size, total_rows - done)
            part = Part(start_offset=offset)
            while part.rows < group_total:
                want = min(group_total - part.rows, chunk_limit)
                chunk = read_chunk(fh, offset, want, w, b, part, chunk_limit)
                if chunk.rows == 0:
                    raise InputError("参考训练读取行数不足")
                offset = chunk.offset_after
                if on_chunk is not None:
                    on_chunk({"rows_in_chunk": chunk.rows,
                              "rows_this_group": part.rows,
                              "group_rows": group_total, "offset_after": offset})
            grad_w = 2.0 * part.gw / part.rows
            grad_b = 2.0 * part.gb / part.rows
            v_w = momentum * v_w - lr * grad_w
            v_b = momentum * v_b - lr * grad_b
            w += v_w
            b += v_b
            done += part.rows
            updates += 1
    return {
        "w": w, "b": b, "v_w": v_w, "v_b": v_b,
        "updates": updates, "samples": done,
    }


def audit(conn: sqlite3.Connection, run_id: int) -> dict:
    """Replay the update chain and check parameters, momentum and progress agree."""
    run = get_run(conn, run_id)
    rows = conn.execute(
        "SELECT * FROM updates WHERE run_id=? ORDER BY idx", (run_id,)
    ).fetchall()
    w, b, v_w, v_b = run["init_w"], run["init_b"], 0.0, 0.0
    problems: list[str] = []
    samples = 0
    prev_offset = int(run["data_start"])
    for expect, rec in enumerate(rows):
        if rec["idx"] != expect:
            problems.append(f"更新序号不连续：期望 {expect}，实际 {rec['idx']}")
        if rec["offset_before"] != prev_offset:
            problems.append(f"更新 {rec['idx']} 的起始偏移与上次结束偏移不衔接")
        prev_offset = int(rec["offset_after"])
        samples += int(rec["samples"])
        v_w = run["momentum"] * v_w - run["lr"] * rec["grad_w"]
        v_b = run["momentum"] * v_b - run["lr"] * rec["grad_b"]
        w += v_w
        b += v_b
        if not (w == rec["w_after"] and b == rec["b_after"]):
            problems.append(f"更新 {rec['idx']} 的参数与审计记录不一致")
        if not (v_w == rec["v_w"] and v_b == rec["v_b"]):
            problems.append(f"更新 {rec['idx']} 的动量与审计记录不一致")
    if samples != run["samples_done"]:
        problems.append(f"样本累计不一致：审计 {samples}，进度 {run['samples_done']}")
    if len(rows) != run["updates_done"]:
        problems.append(f"更新数不一致：审计 {len(rows)}，进度 {run['updates_done']}")
    if not (w == run["w"] and b == run["b"]):
        problems.append("最终参数与进度记录不一致")
    if not (v_w == run["v_w"] and v_b == run["v_b"]):
        problems.append("最终动量与进度记录不一致")
    part_rows = int(run["part_rows"])
    if part_rows < 0 or part_rows >= max(1, int(run["group_size"])):
        problems.append("未成组的累计行数超出每组样本数")
    if samples + part_rows > run["total_rows"]:
        problems.append("已计入样本 + 未成组行数超过总行数")
    if part_rows == 0:
        if run["next_offset"] != prev_offset:
            problems.append("next_offset 与最后一条更新记录不一致")
        if int(run["part_gw"]) != 0 or int(run["part_loss"]) != 0:
            problems.append("未成组累计应为空")
    else:
        if run["next_offset"] == prev_offset:
            problems.append("存在未成组累计但读取偏移未前进")
        if int(run["part_start_offset"]) != prev_offset:
            problems.append("未成组的起始偏移与上次结束偏移不衔接")
    if run["status"] == "finished":
        if samples != run["total_rows"] or part_rows != 0:
            problems.append("已结束但样本数不等于总行数")
        if len(rows) != (run["total_rows"] + run["group_size"] - 1) // run["group_size"]:
            problems.append("已结束但更新次数不等于组数")
    return {
        "ok": not problems,
        "problems": problems,
        "updates": len(rows),
        "samples": samples,
        "part_rows": part_rows,
        "w": w,
        "b": b,
        "v_w": v_w,
        "v_b": v_b,
        "run": run,
    }
