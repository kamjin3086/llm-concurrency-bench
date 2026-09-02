"""SQLite persistence for benchmark jobs and historical reports."""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any


SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS sessions (
  id TEXT PRIMARY KEY, token_hash TEXT UNIQUE NOT NULL, csrf TEXT NOT NULL,
  created_at REAL NOT NULL, expires_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS endpoints (
  id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, base_url TEXT NOT NULL,
  kind TEXT NOT NULL DEFAULT 'auto', lemonade_url TEXT, api_key_env TEXT,
  enabled INTEGER NOT NULL DEFAULT 1, created_at REAL NOT NULL, updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS presets (
  id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT UNIQUE NOT NULL, description TEXT,
  config_json TEXT NOT NULL, is_default INTEGER NOT NULL DEFAULT 0,
  created_at REAL NOT NULL, updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS jobs (
  id TEXT PRIMARY KEY, status TEXT NOT NULL, config_json TEXT NOT NULL,
  created_at REAL NOT NULL, started_at REAL, finished_at REAL,
  progress_json TEXT NOT NULL DEFAULT '{}', error TEXT
);
CREATE TABLE IF NOT EXISTS runs (
  id TEXT PRIMARY KEY, job_id TEXT NOT NULL REFERENCES jobs(id), model_id TEXT NOT NULL,
  model_label TEXT NOT NULL, status TEXT NOT NULL, started_at REAL, finished_at REAL,
  request_snapshot_json TEXT NOT NULL DEFAULT '{}', runtime_snapshot_json TEXT NOT NULL DEFAULT '{}',
  system_snapshot_json TEXT NOT NULL DEFAULT '{}', summary_json TEXT NOT NULL DEFAULT '{}',
  error TEXT
);
CREATE TABLE IF NOT EXISTS batches (
  id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL REFERENCES runs(id),
  concurrency INTEGER NOT NULL, repetition INTEGER NOT NULL, status TEXT NOT NULL,
  summary_json TEXT NOT NULL DEFAULT '{}', created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS requests (
  id INTEGER PRIMARY KEY AUTOINCREMENT, batch_id INTEGER NOT NULL REFERENCES batches(id),
  worker INTEGER NOT NULL, result_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_runs_job ON runs(job_id);
CREATE INDEX IF NOT EXISTS idx_batches_run ON batches(run_id);
"""


def now() -> float:
    return time.time()


class Store:
    def __init__(self, path: str | Path):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.init()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def init(self) -> None:
        with self.connect() as conn:
            conn.executescript(SCHEMA)
            if not conn.execute("SELECT 1 FROM settings WHERE key='password_hash'").fetchone():
                conn.execute("INSERT INTO settings(key,value) VALUES('password_hash','')")
            if not conn.execute("SELECT 1 FROM endpoints").fetchone():
                stamp = now()
                conn.execute(
                    "INSERT INTO endpoints(name,base_url,kind,lemonade_url,created_at,updated_at) VALUES(?,?,?,?,?,?)",
                    ("llama-swap local", "http://127.0.0.1:8101/v1", "llama-swap", "http://127.0.0.1:13305", stamp, stamp),
                )
            if not conn.execute("SELECT 1 FROM presets").fetchone():
                stamp = now()
                presets = [
                    ("快速检查", "新增模型后的快速真实生成检查", {"concurrencies": [1, 4], "repetitions": 1, "max_tokens": 128, "warmup": 0, "preload": True}),
                    ("标准测试", "日常可比的并发吞吐测试", {"concurrencies": [1, 2, 3, 4], "repetitions": 2, "max_tokens": 512, "warmup": 1, "preload": True}),
                    ("稳定性测试", "更多重复以观察波动和失败率", {"concurrencies": [1, 2, 3, 4], "repetitions": 5, "max_tokens": 1024, "warmup": 1, "preload": True}),
                ]
                for index, (name, description, config) in enumerate(presets):
                    conn.execute(
                        "INSERT INTO presets(name,description,config_json,is_default,created_at,updated_at) VALUES(?,?,?,?,?,?)",
                        (name, description, json.dumps(config), int(index == 1), stamp, stamp),
                    )

    def setting(self, key: str, default: str = "") -> str:
        with self.connect() as conn:
            row = conn.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
            return str(row[0]) if row else default

    def set_setting(self, key: str, value: str) -> None:
        with self.connect() as conn:
            conn.execute("INSERT INTO settings(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))

    def create_session(self, token_hash: str, csrf: str, days: int = 14) -> str:
        session_id = uuid.uuid4().hex
        with self.connect() as conn:
            conn.execute("INSERT INTO sessions VALUES(?,?,?,?,?)", (session_id, token_hash, csrf, now(), now() + days * 86400))
        return session_id

    def session(self, token_hash: str) -> sqlite3.Row | None:
        with self.connect() as conn:
            return conn.execute("SELECT * FROM sessions WHERE token_hash=? AND expires_at>?", (token_hash, now())).fetchone()

    def delete_session(self, token_hash: str) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM sessions WHERE token_hash=?", (token_hash,))

    def endpoints(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            return [dict(row) for row in conn.execute("SELECT * FROM endpoints WHERE enabled=1 ORDER BY id")]

    def endpoint(self, endpoint_id: int) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM endpoints WHERE id=?", (endpoint_id,)).fetchone()
            return dict(row) if row else None

    def upsert_endpoint(self, value: dict[str, Any]) -> dict[str, Any]:
        stamp = now()
        with self.connect() as conn:
            if value.get("id"):
                conn.execute(
                    "UPDATE endpoints SET name=?,base_url=?,kind=?,lemonade_url=?,api_key_env=?,updated_at=? WHERE id=?",
                    (value.get("name") or value.get("base_url"), value["base_url"], value.get("kind", "auto"), value.get("lemonade_url"), value.get("api_key_env"), stamp, int(value["id"])),
                )
                row = conn.execute("SELECT * FROM endpoints WHERE id=?", (int(value["id"]),)).fetchone()
            else:
                cur = conn.execute(
                    "INSERT INTO endpoints(name,base_url,kind,lemonade_url,api_key_env,created_at,updated_at) VALUES(?,?,?,?,?,?,?)",
                    (value.get("name") or value["base_url"], value["base_url"], value.get("kind", "auto"), value.get("lemonade_url"), value.get("api_key_env"), stamp, stamp),
                )
                row = conn.execute("SELECT * FROM endpoints WHERE id=?", (cur.lastrowid,)).fetchone()
            return dict(row)

    def presets(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            result = []
            for row in conn.execute("SELECT * FROM presets ORDER BY is_default DESC,id"):
                item = dict(row)
                item["config"] = json.loads(item.pop("config_json"))
                result.append(item)
            return result

    def preset(self, preset_id: int) -> dict[str, Any] | None:
        return next((item for item in self.presets() if item["id"] == int(preset_id)), None)

    def upsert_preset(self, value: dict[str, Any]) -> dict[str, Any]:
        stamp = now()
        config = value.get("config") or {}
        with self.connect() as conn:
            if value.get("id"):
                conn.execute("UPDATE presets SET name=?,description=?,config_json=?,updated_at=? WHERE id=?", (value["name"], value.get("description", ""), json.dumps(config), stamp, int(value["id"])))
                row = conn.execute("SELECT * FROM presets WHERE id=?", (int(value["id"]),)).fetchone()
            else:
                cur = conn.execute("INSERT INTO presets(name,description,config_json,created_at,updated_at) VALUES(?,?,?,?,?)", (value["name"], value.get("description", ""), json.dumps(config), stamp, stamp))
                row = conn.execute("SELECT * FROM presets WHERE id=?", (cur.lastrowid,)).fetchone()
            item = dict(row)
            item["config"] = json.loads(item.pop("config_json"))
            return item

    def create_job(self, config: dict[str, Any]) -> str:
        job_id = uuid.uuid4().hex
        with self.connect() as conn:
            conn.execute("INSERT INTO jobs(id,status,config_json,created_at,progress_json) VALUES(?,?,?,?,?)", (job_id, "queued", json.dumps(config, ensure_ascii=False), now(), json.dumps({"phase": "queued"})))
        return job_id

    def job(self, job_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            if not row:
                return None
            item = dict(row)
            item["config"] = json.loads(item.pop("config_json"))
            item["progress"] = json.loads(item.pop("progress_json") or "{}")
            return item

    def jobs(self, limit: int = 100) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
            result = []
            for row in rows:
                item = dict(row)
                item["config"] = json.loads(item.pop("config_json"))
                item["progress"] = json.loads(item.pop("progress_json") or "{}")
                item["runs"] = self._runs_conn(conn, item["id"], detail=False)
                result.append(item)
            return result

    def update_job(self, job_id: str, *, status: str | None = None, progress: dict[str, Any] | None = None, error: str | None = None, started: bool = False, finished: bool = False) -> None:
        fields: list[str] = []
        values: list[Any] = []
        if status is not None:
            fields.append("status=?"); values.append(status)
        if progress is not None:
            fields.append("progress_json=?"); values.append(json.dumps(progress, ensure_ascii=False))
        if error is not None:
            fields.append("error=?"); values.append(error)
        if started:
            fields.append("started_at=?"); values.append(now())
        if finished:
            fields.append("finished_at=?"); values.append(now())
        if fields:
            values.append(job_id)
            with self.connect() as conn:
                conn.execute(f"UPDATE jobs SET {','.join(fields)} WHERE id=?", values)

    def create_run(self, job_id: str, model_id: str, model_label: str, request_snapshot: dict[str, Any], system_snapshot: dict[str, Any]) -> str:
        run_id = uuid.uuid4().hex
        with self.connect() as conn:
            conn.execute("INSERT INTO runs(id,job_id,model_id,model_label,status,started_at,request_snapshot_json,system_snapshot_json) VALUES(?,?,?,?,?,?,?,?)", (run_id, job_id, model_id, model_label, "running", now(), json.dumps(request_snapshot, ensure_ascii=False), json.dumps(system_snapshot, ensure_ascii=False)))
        return run_id

    def update_run(self, run_id: str, *, status: str | None = None, runtime: dict[str, Any] | None = None, summary: dict[str, Any] | None = None, error: str | None = None, finished: bool = False) -> None:
        fields: list[str] = []
        values: list[Any] = []
        if status is not None:
            fields.append("status=?"); values.append(status)
        if runtime is not None:
            fields.append("runtime_snapshot_json=?"); values.append(json.dumps(runtime, ensure_ascii=False))
        if summary is not None:
            fields.append("summary_json=?"); values.append(json.dumps(summary, ensure_ascii=False))
        if error is not None:
            fields.append("error=?"); values.append(error)
        if finished:
            fields.append("finished_at=?"); values.append(now())
        if fields:
            values.append(run_id)
            with self.connect() as conn:
                conn.execute(f"UPDATE runs SET {','.join(fields)} WHERE id=?", values)

    def add_batch(self, run_id: str, concurrency: int, repetition: int, summary: dict[str, Any], results: list[dict[str, Any]]) -> int:
        with self.connect() as conn:
            cur = conn.execute("INSERT INTO batches(run_id,concurrency,repetition,status,summary_json,created_at) VALUES(?,?,?,?,?,?)", (run_id, concurrency, repetition, "completed", json.dumps(summary, ensure_ascii=False), now()))
            batch_id = int(cur.lastrowid)
            conn.executemany("INSERT INTO requests(batch_id,worker,result_json) VALUES(?,?,?)", [(batch_id, int(row.get("worker", 0)), json.dumps(row, ensure_ascii=False)) for row in results])
            return batch_id

    def _runs_conn(self, conn: sqlite3.Connection, job_id: str, detail: bool = True) -> list[dict[str, Any]]:
        runs = []
        for row in conn.execute("SELECT * FROM runs WHERE job_id=? ORDER BY rowid", (job_id,)):
            item = dict(row)
            for key in ("request_snapshot_json", "runtime_snapshot_json", "system_snapshot_json", "summary_json"):
                item[key[:-5] if key.endswith("_json") else key] = json.loads(item.pop(key) or "{}")
            if detail:
                batches = []
                for batch in conn.execute("SELECT * FROM batches WHERE run_id=? ORDER BY concurrency,repetition", (item["id"],)):
                    b = dict(batch); b["summary"] = json.loads(b.pop("summary_json") or "{}"); b["requests"] = [json.loads(x[0]) for x in conn.execute("SELECT result_json FROM requests WHERE batch_id=? ORDER BY worker", (batch["id"],))]; batches.append(b)
                item["batches"] = batches
            runs.append(item)
        return runs

    def job_detail(self, job_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            if not row:
                return None
            item = dict(row); item["config"] = json.loads(item.pop("config_json")); item["progress"] = json.loads(item.pop("progress_json") or "{}"); item["runs"] = self._runs_conn(conn, job_id, detail=True); return item

    def mark_interrupted(self) -> None:
        with self.connect() as conn:
            conn.execute("UPDATE jobs SET status='interrupted',finished_at=?,error='服务重启时任务未完成' WHERE status='running'", (now(),))
            conn.execute("UPDATE runs SET status='interrupted',finished_at=?,error='服务重启时任务未完成' WHERE status='running'", (now(),))
