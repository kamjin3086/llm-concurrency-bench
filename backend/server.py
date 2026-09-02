"""Local/LAN web server for the benchmark workbench.

The server deliberately uses the Python standard library for persistence and HTTP
so the project is easy to install on the machine that runs the models.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import secrets
import signal
import statistics
import threading
import time
import urllib.parse
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .engine import capture_runtime, model_list, preload, run_batch
from .metrics import sweet_spot
from .security import hash_password, json_safe, new_token, verify_password, redact
from .store import Store


ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
DEFAULT_DB = Path(os.environ.get("LLM_BENCH_DB", Path.home() / ".local/share/llm-concurrency-bench/bench.db"))


def _json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _safe_url(value: str) -> str:
    parsed = urllib.parse.urlsplit(str(value).strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username or parsed.password:
        raise ValueError("URL 必须是 http(s) 地址，且不能包含用户名或密码")
    return str(value).rstrip("/")


class JobManager:
    def __init__(self, store: Store):
        self.store = store
        self.stop = threading.Event()
        self.wake = threading.Event()
        self.cancel_events: dict[str, threading.Event] = {}
        self.subscribers: dict[str, set[queue.Queue[dict[str, Any]]]] = {}
        self.lock = threading.RLock()
        self.store.mark_interrupted()
        self.thread = threading.Thread(target=self._loop, name="benchmark-queue", daemon=True)
        self.thread.start()

    def submit(self, config: dict[str, Any]) -> str:
        job_id = self.store.create_job(config)
        with self.lock:
            self.cancel_events[job_id] = threading.Event()
        self.wake.set()
        self.emit(job_id, "queued", {"position": self.queue_position(job_id)})
        return job_id

    def queue_position(self, job_id: str) -> int:
        queued = [job["id"] for job in reversed(self.store.jobs(1000)) if job["status"] == "queued"]
        try:
            return queued.index(job_id) + 1
        except ValueError:
            return 0

    def cancel(self, job_id: str) -> bool:
        with self.lock:
            event = self.cancel_events.get(job_id)
        job = self.store.job(job_id)
        if not job or job["status"] in {"completed", "failed", "cancelled", "interrupted"}:
            return False
        if event:
            event.set()
        if job["status"] == "queued":
            self.store.update_job(job_id, status="cancelled", progress={"phase": "cancelled"}, finished=True)
            self.emit(job_id, "job_status", {"status": "cancelled"})
        return True

    def subscribe(self, job_id: str) -> tuple[queue.Queue[dict[str, Any]], callable]:
        channel: queue.Queue[dict[str, Any]] = queue.Queue()
        with self.lock:
            self.subscribers.setdefault(job_id, set()).add(channel)
        job = self.store.job(job_id)
        if job:
            channel.put({"event": "snapshot", "data": {"status": job["status"], "progress": job["progress"]}})

        def unsubscribe() -> None:
            with self.lock:
                self.subscribers.get(job_id, set()).discard(channel)

        return channel, unsubscribe

    def emit(self, job_id: str, event: str, data: dict[str, Any]) -> None:
        payload = {"event": event, "data": data}
        if event in {"phase", "batch_result", "snapshot", "job_status", "error"}:
            job = self.store.job(job_id)
            if job:
                progress = dict(job.get("progress") or {})
                progress.update(data)
                progress["phase"] = data.get("name", data.get("phase", event))
                self.store.update_job(job_id, progress=progress)
        with self.lock:
            channels = list(self.subscribers.get(job_id, set()))
        for channel in channels:
            try:
                channel.put_nowait(payload)
            except queue.Full:
                pass

    def _cancel_event(self, job_id: str) -> threading.Event:
        with self.lock:
            return self.cancel_events.setdefault(job_id, threading.Event())

    def _loop(self) -> None:
        while not self.stop.is_set():
            jobs = [job for job in self.store.jobs(1000) if job["status"] == "queued"]
            if not jobs:
                self.wake.wait(1.0)
                self.wake.clear()
                continue
            job = jobs[-1]  # jobs() is newest first; execute oldest queued first.
            try:
                self._run(job)
            except Exception as exc:  # noqa: BLE001 - job failure belongs in history
                self.store.update_job(job["id"], status="failed", error=repr(exc), progress={"phase": "failed", "error": repr(exc)}, finished=True)
                self.emit(job["id"], "job_status", {"status": "failed", "error": repr(exc)})

    def _run(self, job: dict[str, Any]) -> None:
        job_id = job["id"]
        cancel = self._cancel_event(job_id)
        config = dict(job["config"])
        self.store.update_job(job_id, status="running", started=True, progress={"phase": "starting", "models_total": len(config.get("models", [])), "models_done": 0})
        self.emit(job_id, "job_status", {"status": "running"})
        runs: list[dict[str, Any]] = []
        models = config.get("models") or []
        for index, model in enumerate(models):
            if cancel.is_set():
                break
            model_id = str(model.get("id"))
            model_label = str(model.get("label") or model_id)
            self.emit(job_id, "model_start", {"model_index": index, "models_total": len(models), "model_id": model_id, "model_label": model_label})
            request_snapshot = {"endpoint": config.get("base_url"), "model": model_id, "preset": config.get("preset_name"), "config": config.get("benchmark", {})}
            run_id = self.store.create_run(job_id, model_id, model_label, request_snapshot, {})
            try:
                self.emit(job_id, "phase", {"model_id": model_id, "name": "preload", "message": "等待真实 completion"})
                api_key = os.environ.get(config.get("api_key_env") or "") or None
                preload_info = preload({**config.get("benchmark", {}), "base_url": config.get("base_url"), "api_key": api_key}, model_id, lambda event, data: self.emit(job_id, event, {"model_id": model_id, **data}), cancel)
                runtime = capture_runtime({**config, "base_url": config.get("base_url")}, model_id, lambda event, data: self.emit(job_id, event, {"model_id": model_id, **data}))
                runtime["preload"] = preload_info
                self.store.update_run(run_id, runtime=runtime)
                bench = {**config.get("benchmark", {}), "base_url": config.get("base_url"), "api_key": api_key}
                warmup = int(bench.get("warmup", 1))
                for warm in range(warmup):
                    if cancel.is_set():
                        raise RuntimeError("cancelled")
                    self.emit(job_id, "phase", {"model_id": model_id, "name": "warmup", "current": warm + 1, "total": warmup})
                    run_batch(bench, model_id, 1, lambda _event, _data: None, cancel)
                rows_by_c: dict[int, list[dict[str, Any]]] = {}
                concurrencies = [int(value) for value in bench.get("concurrencies", [1, 2, 3, 4])]
                repetitions = int(bench.get("repetitions", 2))
                total_batches = len(concurrencies) * repetitions
                completed_batches = 0
                for concurrency in concurrencies:
                    for repetition in range(1, repetitions + 1):
                        if cancel.is_set():
                            raise RuntimeError("cancelled")
                        self.emit(job_id, "phase", {"model_id": model_id, "name": "benchmark", "concurrency": concurrency, "repetition": repetition, "total_batches": total_batches, "completed_batches": completed_batches})
                        summary, request_rows = run_batch(bench, model_id, concurrency, lambda event, data: self.emit(job_id, event, {"model_id": model_id, "run_id": run_id, "repetition": repetition, **data}), cancel)
                        rows_by_c.setdefault(concurrency, []).append(summary)
                        self.store.add_batch(run_id, concurrency, repetition, summary, request_rows)
                        completed_batches += 1
                aggregate = []
                for concurrency, summaries in rows_by_c.items():
                    # The UI needs one stable point per concurrency; retain all repetitions in batches.
                    keys = ("prompt_tokens", "completion_tokens", "wall_s", "aggregate_e2e_tps", "aggregate_decode_tps", "avg_stream_decode_tps", "ttft_avg_s", "ttft_p50_s", "ttft_p95_s", "ttft_max_s", "failed_requests")
                    point = {"concurrency": concurrency, "repetitions": len(summaries)}
                    for key in keys:
                        values = [float(item[key]) for item in summaries if item.get(key) is not None]
                        point[key] = statistics.mean(values) if values else None
                    point["failed_requests"] = int(sum(int(item.get("failed_requests") or 0) for item in summaries))
                    aggregate.append(point)
                summary = {"points": sorted(aggregate, key=lambda item: item["concurrency"]), "sweet_spot": sweet_spot(aggregate), "total_batches": total_batches}
                self.store.update_run(run_id, status="completed", summary=summary, finished=True)
                runs.append({"id": run_id, "model_id": model_id, "model_label": model_label, "summary": summary})
                self.emit(job_id, "model_done", {"model_id": model_id, "model_label": model_label, "run_id": run_id, "summary": summary, "models_done": index + 1})
            except Exception as exc:
                status = "cancelled" if cancel.is_set() else "failed"
                self.store.update_run(run_id, status=status, error=repr(exc), finished=True)
                self.emit(job_id, "model_error", {"model_id": model_id, "run_id": run_id, "status": status, "error": repr(exc)})
                if status == "failed":
                    # Continue to the next selected model; a broken model must not erase useful history.
                    continue
                break
        final_status = "cancelled" if cancel.is_set() else ("completed" if runs else "failed")
        self.store.update_job(job_id, status=final_status, finished=True, progress={"phase": final_status, "models_done": len(runs), "models_total": len(models)})
        self.emit(job_id, "job_status", {"status": final_status, "models_done": len(runs), "models_total": len(models)})


class App:
    def __init__(self, db_path: str | Path):
        self.store = Store(db_path)
        self.jobs = JobManager(self.store)

    def setup_required(self) -> bool:
        return not bool(self.store.setting("password_hash"))

    def session(self, handler: BaseHTTPRequestHandler) -> tuple[Any, str | None]:
        cookie = SimpleCookie(handler.headers.get("Cookie", ""))
        value = cookie.get("bench_sid")
        if not value:
            return None, None
        token = value.value
        row = self.store.session(_sha(token))
        return row, token

    def require(self, handler: BaseHTTPRequestHandler, csrf: bool = False) -> Any:
        row, _ = self.session(handler)
        if not row:
            raise ApiError(HTTPStatus.UNAUTHORIZED, "需要登录")
        if csrf and not secrets.compare_digest(str(row["csrf"]), handler.headers.get("X-CSRF-Token", "")):
            raise ApiError(HTTPStatus.FORBIDDEN, "CSRF token 无效")
        return row


class ApiError(Exception):
    def __init__(self, status: HTTPStatus, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


class Handler(BaseHTTPRequestHandler):
    server_version = "BenchRoom/0.1"

    @property
    def app(self) -> App:
        return self.server.app  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: Any) -> None:
        # Keep request logs useful without ever echoing request bodies or credentials.
        print(f"[{time.strftime('%H:%M:%S')}] {self.command} {self.path.split('?')[0]} - {fmt % args}", flush=True)

    def _send(self, status: int, body: bytes, content_type: str = "application/json; charset=utf-8", headers: dict[str, str] | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, value: Any, headers: dict[str, str] | None = None) -> None:
        self._send(status, _json(value), headers=headers)

    def _read_json(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 5_000_000:
                raise ApiError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "请求过大")
            value = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(value, dict):
                raise ValueError
            return value
        except ApiError:
            raise
        except Exception as exc:
            raise ApiError(HTTPStatus.BAD_REQUEST, f"JSON 无效: {exc}") from exc

    def _serve_static(self, path: str) -> None:
        relative = "index.html" if path in {"", "/"} else path.removeprefix("/")
        target = (WEB / relative).resolve()
        if WEB not in target.parents and target != WEB:
            raise ApiError(HTTPStatus.NOT_FOUND, "not found")
        if not target.is_file():
            target = WEB / "index.html"
        content_types = {".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".svg": "image/svg+xml"}
        self._send(HTTPStatus.OK, target.read_bytes(), content_types.get(target.suffix, "application/octet-stream"), {"Cache-Control": "no-cache"})

    def _api_path(self) -> tuple[str, list[str], dict[str, list[str]]]:
        parsed = urllib.parse.urlsplit(self.path)
        parts = [urllib.parse.unquote(part) for part in parsed.path.split("/") if part]
        query = urllib.parse.parse_qs(parsed.query)
        return parsed.path, parts, query

    def do_GET(self) -> None:  # noqa: N802
        try:
            path, parts, query = self._api_path()
            if not path.startswith("/api/"):
                return self._serve_static(path)
            if parts == ["api", "session"]:
                row, _ = self.app.session(self)
                self._json(200, {"setup_required": self.app.setup_required(), "authenticated": bool(row), "csrf": row["csrf"] if row else None})
                return
            self.app.require(self)
            if parts == ["api", "endpoints"]:
                self._json(200, {"endpoints": self.app.store.endpoints()}); return
            if parts == ["api", "presets"]:
                self._json(200, {"presets": self.app.store.presets()}); return
            if parts == ["api", "models"]:
                endpoint_id = int((query.get("endpoint_id") or ["1"])[0]); endpoint = self.app.store.endpoint(endpoint_id)
                if not endpoint:
                    raise ApiError(HTTPStatus.NOT_FOUND, "端点不存在")
                api_key = os.environ.get(endpoint.get("api_key_env") or "") or None
                models = model_list(endpoint["base_url"], api_key=api_key)
                health = self._health_for(endpoint)
                loaded = {item.get("model_name") for item in (health or {}).get("all_models_loaded", [])}
                for item in models:
                    item["loaded"] = item.get("id") in loaded
                self._json(200, {"endpoint": endpoint, "models": models, "health": health}); return
            if parts == ["api", "health"]:
                endpoint_id = int((query.get("endpoint_id") or ["1"])[0]); endpoint = self.app.store.endpoint(endpoint_id)
                if not endpoint: raise ApiError(HTTPStatus.NOT_FOUND, "端点不存在")
                self._json(200, {"endpoint": endpoint, "health": self._health_for(endpoint)}); return
            if parts == ["api", "history"]:
                self._json(200, {"jobs": self.app.store.jobs(100)}); return
            if len(parts) == 3 and parts[:2] == ["api", "jobs"]:
                detail = self.app.store.job_detail(parts[2])
                if not detail: raise ApiError(HTTPStatus.NOT_FOUND, "任务不存在")
                self._json(200, detail); return
            if len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[3] == "share":
                detail = self.app.store.job_detail(parts[2])
                if not detail: raise ApiError(HTTPStatus.NOT_FOUND, "任务不存在")
                self._json(200, redact(detail, remove_prompts=True)); return
            if len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[3] == "export":
                detail = self.app.store.job_detail(parts[2])
                if not detail: raise ApiError(HTTPStatus.NOT_FOUND, "任务不存在")
                self._send(200, json_safe(detail).encode("utf-8"), "application/json; charset=utf-8", {"Content-Disposition": f"attachment; filename=bench-{parts[2]}.json"}); return
            if len(parts) == 4 and parts[:2] == ["api", "events"] and parts[3] == "stream":
                return self._events(parts[2])
            raise ApiError(HTTPStatus.NOT_FOUND, "API 不存在")
        except ApiError as exc:
            self._json(int(exc.status), {"error": exc.message})
        except Exception as exc:  # noqa: BLE001 - return a readable API error
            self._json(500, {"error": repr(exc)})

    def _health_for(self, endpoint: dict[str, Any]) -> dict[str, Any] | None:
        root = str(endpoint["base_url"]).rstrip("/")
        if root.endswith("/v1"): root = root[:-3]
        for path in ("/api/v1/health", "/v1/health", "/health"):
            try:
                from .engine import get_json
                return get_json(root + path, 3)
            except Exception:
                continue
        return None

    def _events(self, job_id: str) -> None:
        if not self.app.store.job(job_id):
            raise ApiError(HTTPStatus.NOT_FOUND, "任务不存在")
        channel, unsubscribe = self.app.jobs.subscribe(job_id)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        try:
            while True:
                try:
                    item = channel.get(timeout=15)
                    self.wfile.write(f"event: {item['event']}\ndata: {json.dumps(item['data'], ensure_ascii=False)}\n\n".encode("utf-8"))
                    self.wfile.flush()
                    if item["event"] == "job_status" and item["data"].get("status") in {"completed", "failed", "cancelled", "interrupted"}:
                        break
                except queue.Empty:
                    self.wfile.write(b": heartbeat\n\n"); self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            unsubscribe()

    def do_POST(self) -> None:  # noqa: N802
        try:
            path, parts, _query = self._api_path()
            data = self._read_json()
            if parts == ["api", "setup"]:
                if not self.app.setup_required(): raise ApiError(HTTPStatus.CONFLICT, "已完成初始化")
                password = str(data.get("password", ""))
                try:
                    password_hash = hash_password(password)
                except ValueError as exc:
                    raise ApiError(HTTPStatus.BAD_REQUEST, str(exc)) from exc
                self.app.store.set_setting("password_hash", password_hash)
                self._login_response(); return
            if parts == ["api", "login"]:
                if self.app.setup_required(): raise ApiError(HTTPStatus.CONFLICT, "请先设置密码")
                if not verify_password(str(data.get("password", "")), self.app.store.setting("password_hash")):
                    raise ApiError(HTTPStatus.UNAUTHORIZED, "密码错误")
                self._login_response(); return
            if parts == ["api", "logout"]:
                self.app.require(self, csrf=True); row, token = self.app.session(self)
                if token: self.app.store.delete_session(_sha(token))
                self._json(200, {"ok": True}, {"Set-Cookie": "bench_sid=; Max-Age=0; Path=/; HttpOnly; SameSite=Lax"}); return
            self.app.require(self, csrf=True)
            if parts == ["api", "endpoints"]:
                data["base_url"] = _safe_url(data.get("base_url", ""));
                if data.get("lemonade_url"): data["lemonade_url"] = _safe_url(data["lemonade_url"])
                self._json(200, self.app.store.upsert_endpoint(data)); return
            if parts == ["api", "presets"]:
                if not str(data.get("name", "")).strip(): raise ApiError(400, "预设名称不能为空")
                self._json(200, self.app.store.upsert_preset(data)); return
            if parts == ["api", "jobs"]:
                job_config = self._validate_job(data)
                job_id = self.app.jobs.submit(job_config)
                self._json(202, {"id": job_id, "status": "queued"}); return
            if len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[3] == "cancel":
                if not self.app.jobs.cancel(parts[2]): raise ApiError(409, "任务已结束或不存在")
                self._json(200, {"ok": True}); return
            raise ApiError(HTTPStatus.NOT_FOUND, "API 不存在")
        except ApiError as exc:
            self._json(int(exc.status), {"error": exc.message})
        except Exception as exc:  # noqa: BLE001
            self._json(500, {"error": repr(exc)})

    def _login_response(self) -> None:
        token = new_token(); csrf = new_token(); self.app.store.create_session(_sha(token), csrf)
        self._json(200, {"authenticated": True, "csrf": csrf}, {"Set-Cookie": f"bench_sid={token}; Max-Age=1209600; Path=/; HttpOnly; SameSite=Lax"})

    def _validate_job(self, data: dict[str, Any]) -> dict[str, Any]:
        endpoint_id = int(data.get("endpoint_id", 1)); endpoint = self.app.store.endpoint(endpoint_id)
        if not endpoint: raise ApiError(400, "端点不存在")
        models = data.get("models")
        if not isinstance(models, list) or not models: raise ApiError(400, "至少选择一个模型")
        clean_models = []
        for model in models:
            if not isinstance(model, dict) or not str(model.get("id", "")).strip(): raise ApiError(400, "模型选择无效")
            clean_models.append({"id": str(model["id"]), "label": str(model.get("label") or model["id"])})
        benchmark = dict(data.get("benchmark") or {})
        benchmark["concurrencies"] = [int(value) for value in benchmark.get("concurrencies", [1, 2, 3, 4]) if 1 <= int(value) <= 128]
        if not benchmark["concurrencies"]: raise ApiError(400, "至少选择一个并发档位")
        benchmark["repetitions"] = max(1, min(100, int(benchmark.get("repetitions", 2))))
        benchmark["max_tokens"] = max(1, min(131072, int(benchmark.get("max_tokens", 512))))
        benchmark["warmup"] = max(0, min(10, int(benchmark.get("warmup", 1))))
        benchmark["timeout"] = max(5, min(7200, float(benchmark.get("timeout", 900))))
        benchmark["temperature"] = float(benchmark.get("temperature", 0.0))
        benchmark["seed"] = int(benchmark.get("seed", 42))
        benchmark["unique_prompts"] = bool(benchmark.get("unique_prompts", True))
        return {"endpoint_id": endpoint_id, "base_url": endpoint["base_url"], "lemonade_url": endpoint.get("lemonade_url"), "models": clean_models, "preset_id": data.get("preset_id"), "preset_name": str(data.get("preset_name") or "自定义"), "benchmark": benchmark, "api_key_env": endpoint.get("api_key_env")}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Browser benchmark workbench")
    parser.add_argument("--host", default=os.environ.get("LLM_BENCH_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("LLM_BENCH_PORT", "8790")))
    parser.add_argument("--db", default=str(DEFAULT_DB))
    args = parser.parse_args(argv)
    app = App(args.db)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.app = app  # type: ignore[attr-defined]
    print(f"BenchRoom listening on http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        app.jobs.stop.set(); server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
