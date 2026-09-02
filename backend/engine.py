"""HTTP benchmark engine and runtime parameter evidence collector."""
from __future__ import annotations

import json
import os
import re
import socket
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable

from .metrics import batch_metrics


DEFAULT_SYSTEM = "You are a senior software engineer working as one worker in a parallel coding-agent team."
DEFAULT_PROMPT = (
    "Design and explain a production-quality implementation for a moderately complex software component. "
    "Include architecture, data structures, concurrency/error handling, edge cases, testing strategy, "
    "and representative code. Continue in technical detail until the token budget is exhausted."
)


def endpoint(base: str) -> str:
    b = str(base).rstrip("/")
    return b + "/chat/completions" if b.endswith("/v1") else b + "/v1/chat/completions"


def endpoint_root(base: str) -> str:
    b = str(base).rstrip("/")
    return b[:-3] if b.endswith("/v1") else b


def get_json(url: str, timeout: float = 10.0, headers: dict[str, str] | None = None) -> Any:
    request = urllib.request.Request(url, headers={"Accept": "application/json", **(headers or {})})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8", "replace"))


def model_list(base_url: str, timeout: float = 10.0, api_key: str | None = None) -> list[dict[str, Any]]:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    bases = [str(base_url).rstrip("/")]
    if bases[0].endswith("/v1"):
        bases.append(bases[0][:-3])
    urls = []
    for base in bases:
        urls.append(base + "/v1/models" if not base.endswith("/v1") else base + "/models")
        urls.append(base + "/models")
    seen: set[str] = set()
    last: Exception | None = None
    for url in urls:
        if url in seen:
            continue
        seen.add(url)
        try:
            value = get_json(url, timeout, headers)
            data = value.get("data", []) if isinstance(value, dict) else []
            if data:
                return [item for item in data if isinstance(item, dict) and item.get("id")]
        except Exception as exc:  # noqa: BLE001 - endpoint probing intentionally tries alternatives
            last = exc
    raise RuntimeError(f"cannot discover models from {base_url}: {last}")


def _has_token(chunk: dict[str, Any]) -> bool:
    for choice in chunk.get("choices") or []:
        delta = choice.get("delta") or {}
        if any(isinstance(delta.get(key), str) and delta.get(key) for key in ("content", "reasoning_content", "reasoning")):
            return True
        if delta.get("tool_calls"):
            return True
    return False


@dataclass
class RequestResult:
    ok: bool
    worker: int
    started_s: float
    first_token_s: float | None
    finished_s: float
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str | None
    error: str | None = None
    chunks: int = 0

    def json(self) -> dict[str, Any]:
        value = asdict(self)
        value["ttft_s"] = None if self.first_token_s is None else self.first_token_s - self.started_s
        value["decode_s"] = None if self.first_token_s is None else max(1e-9, self.finished_s - self.first_token_s)
        value["stream_decode_tps"] = (
            None
            if not self.completion_tokens or value["decode_s"] is None
            else max(0, self.completion_tokens - 1) / value["decode_s"]
        )
        return value


def _error_text(exc: Exception) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        try:
            body = exc.read(1200).decode("utf-8", "replace")
        except Exception:
            body = ""
        return f"HTTP {exc.code}: {body}".strip()
    return repr(exc)


def request_one(ep: str, model: str, config: dict[str, Any], worker: int, gate: threading.Barrier, cancel: threading.Event | None = None) -> RequestResult:
    timeout = float(config.get("timeout", 900))
    max_tokens = int(config.get("max_tokens", 512))
    temperature = float(config.get("temperature", 0.0))
    seed = int(config.get("seed", 42)) + worker
    system = str(config.get("system_prompt", DEFAULT_SYSTEM))
    prompt = str(config.get("prompt", DEFAULT_PROMPT))
    if config.get("unique_prompts", True):
        prompt += f"\n\nYou are worker #{worker}. Use scenario variant #{worker} with a distinct module name and independent implementation details."
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": max_tokens,
        "temperature": temperature,
        "seed": seed,
    }
    extra = dict(config.get("extra_body") or {})
    body.update(extra)
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    api_key = config.get("api_key")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(ep, data=json.dumps(body, ensure_ascii=False).encode("utf-8"), method="POST", headers=headers)
    try:
        gate.wait(timeout=30)
    except threading.BrokenBarrierError:
        pass
    started = time.perf_counter()
    first: float | None = None
    prompt_tokens = completion_tokens = chunks = 0
    reason = None
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw in response:
                if cancel and cancel.is_set():
                    break
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                chunks += 1
                if first is None and _has_token(chunk):
                    first = time.perf_counter()
                for choice in chunk.get("choices") or []:
                    if choice.get("finish_reason") is not None:
                        reason = choice.get("finish_reason")
                usage = chunk.get("usage")
                if usage:
                    prompt_tokens = int(usage.get("prompt_tokens") or prompt_tokens or 0)
                    completion_tokens = int(usage.get("completion_tokens") or completion_tokens or 0)
        return RequestResult(True, worker, started, first, time.perf_counter(), prompt_tokens, completion_tokens, reason, chunks=chunks)
    except Exception as exc:  # noqa: BLE001 - failures are part of benchmark output
        return RequestResult(False, worker, started, None, time.perf_counter(), prompt_tokens, completion_tokens, reason, _error_text(exc), chunks)


def preload(config: dict[str, Any], model: str, emit: Callable[[str, dict[str, Any]], None], cancel: threading.Event) -> dict[str, Any]:
    if not config.get("preload", True):
        return {"enabled": False}
    timeout = float(config.get("preload_timeout", 1800))
    retry = float(config.get("preload_retry_interval", 3))
    deadline = time.monotonic() + timeout
    tiny = dict(config)
    tiny.update({
        "max_tokens": int(config.get("preload_max_tokens", 8)),
        "unique_prompts": False,
        "system_prompt": "You are a concise assistant.",
        "prompt": config.get("preload_prompt", "Reply with exactly: READY"),
        "temperature": 0.0,
        "timeout": timeout,
    })
    attempt = 0
    emit("phase", {"name": "preload", "message": "等待模型返回真实 completion（不计入测速）"})
    while not cancel.is_set():
        attempt += 1
        result = request_one(endpoint(config["base_url"]), model, tiny, 1, threading.Barrier(1), cancel)
        if result.ok and result.first_token_s is not None:
            elapsed = result.finished_s - result.started_s
            emit("phase", {"name": "preload_ready", "seconds": elapsed, "ttft_s": result.json().get("ttft_s"), "attempt": attempt})
            settle = float(config.get("settle_seconds", 1.0))
            if settle > 0:
                time.sleep(settle)
            return {"enabled": True, "attempts": attempt, "seconds": elapsed, "ttft_s": result.json().get("ttft_s")}
        if time.monotonic() >= deadline:
            raise RuntimeError(f"model did not become ready within {timeout:.0f}s: {result.error or 'no generated token'}")
        emit("phase", {"name": "preload_retry", "attempt": attempt, "error": result.error or "no generated token"})
        cancel.wait(retry)
    raise RuntimeError("cancelled during preload")


def run_batch(config: dict[str, Any], model: str, concurrency: int, emit: Callable[[str, dict[str, Any]], None], cancel: threading.Event) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if cancel.is_set():
        raise RuntimeError("cancelled")
    gate = threading.Barrier(concurrency)
    results: list[RequestResult] = []
    with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="bench") as executor:
        futures = [executor.submit(request_one, endpoint(config["base_url"]), model, config, i + 1, gate, cancel) for i in range(concurrency)]
        for future in as_completed(futures):
            results.append(future.result())
    rows = [result.json() for result in sorted(results, key=lambda item: item.worker)]
    summary = batch_metrics(rows)
    summary["concurrency"] = concurrency
    emit("batch_result", {"concurrency": concurrency, **summary})
    return summary, rows


def _read_proc_cmdline(pid: int) -> list[str] | None:
    try:
        raw = Path(f"/proc/{int(pid)}/cmdline").read_bytes()
        return [part.decode("utf-8", "replace") for part in raw.split(b"\0") if part]
    except (FileNotFoundError, PermissionError, ValueError):
        return None


def _proc_exe(pid: int) -> str | None:
    try:
        return os.readlink(f"/proc/{int(pid)}/exe")
    except (FileNotFoundError, PermissionError, OSError, ValueError):
        return None


def _proc_start_ticks(pid: int) -> int | None:
    try:
        stat = Path(f"/proc/{int(pid)}/stat").read_text(encoding="utf-8")
        # The comm field may contain spaces; fields after the final ') ' are stable.
        fields = stat.rsplit(") ", 1)[1].split()
        return int(fields[19])  # field 22 overall, index 19 after state
    except (FileNotFoundError, PermissionError, ValueError, IndexError):
        return None


def _pid_listening_ports(pid: int) -> set[int]:
    """Verify a backend PID owns a listening TCP socket without invoking shell tools."""
    inodes: set[str] = set()
    try:
        for fd in Path(f"/proc/{int(pid)}/fd").iterdir():
            try:
                target = os.readlink(fd)
            except OSError:
                continue
            match = re.fullmatch(r"socket:\[(\d+)\]", target)
            if match:
                inodes.add(match.group(1))
    except (FileNotFoundError, PermissionError):
        return set()
    ports: set[int] = set()
    for table in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            lines = Path(table).read_text(encoding="ascii").splitlines()[1:]
        except (FileNotFoundError, PermissionError):
            continue
        for line in lines:
            parts = line.split()
            if len(parts) < 10 or parts[3] != "0A" or parts[9] not in inodes:
                continue
            try:
                ports.add(int(parts[1].split(":", 1)[1], 16))
            except (ValueError, IndexError):
                pass
    return ports


def _health(base_url: str, timeout: float = 5.0) -> tuple[dict[str, Any] | None, str | None]:
    root = endpoint_root(base_url)
    for path in ("/api/v1/health", "/v1/health"):
        try:
            value = get_json(root + path, timeout)
            if isinstance(value, dict):
                return value, root + path
        except Exception:
            continue
    return None, None


def _choose_loaded_model(health: dict[str, Any], requested: str) -> tuple[dict[str, Any] | None, str]:
    models = [item for item in health.get("all_models_loaded", []) if isinstance(item, dict)]
    for item in models:
        if requested in {item.get("model_name"), item.get("id"), item.get("model_loaded")}:
            return item, "requested_id"
    loaded_name = health.get("model_loaded")
    for item in models:
        if item.get("model_name") == loaded_name:
            return item, "health_model_loaded"
    if len(models) == 1:
        return models[0], "single_loaded_model"
    # last_use is the only safe fallback when aliases are transformed by a proxy.
    return max(models, key=lambda item: float(item.get("last_use") or 0), default=None), "latest_last_use_fallback"


def _find_swap_process() -> dict[str, Any] | None:
    """Find the local llama-swap supervisor without assuming a service manager."""
    if not hasattr(os, "getuid"):
        return None
    try:
        own_uid = os.getuid()
        for proc in Path("/proc").iterdir():
            if not proc.name.isdigit():
                continue
            try:
                status = (proc / "status").read_text(encoding="utf-8", errors="replace")
                uid_line = next((line for line in status.splitlines() if line.startswith("Uid:")), "")
                if uid_line.split()[1] != str(own_uid):
                    continue
                argv = _read_proc_cmdline(int(proc.name)) or []
                exe = _proc_exe(int(proc.name)) or ""
                if "llama-swap" not in os.path.basename(exe).lower() and not any("llama-swap" in part.lower() for part in argv):
                    continue
                return {"pid": int(proc.name), "argv": argv, "executable": exe, "start_ticks": _proc_start_ticks(int(proc.name))}
            except (FileNotFoundError, PermissionError, IndexError, ValueError):
                continue
    except (FileNotFoundError, PermissionError):
        return None
    return None


def capture_runtime(config: dict[str, Any], requested_model: str, emit: Callable[[str, dict[str, Any]], None]) -> dict[str, Any]:
    """Capture a structured evidence bundle. Missing optional sources are explicit."""
    evidence: dict[str, Any] = {"captured_at": time.time(), "requested_model": requested_model, "sources": []}
    # Lemonade exposes the structured loaded-model registry on its own API;
    # llama-swap itself normally only exposes /health and /v1/models.
    health, health_url = _health(config.get("lemonade_url") or config.get("base_url", ""))
    if health is None and config.get("lemonade_url"):
        health, health_url = _health(config.get("base_url", ""))
    if health is not None:
        evidence["lemonade_health"] = health
        evidence["sources"].append({"name": "lemonade_health", "url": health_url, "confidence": "authoritative"})
    else:
        evidence["lemonade_health_error"] = "health endpoint unavailable"

    loaded, match_mode = _choose_loaded_model(health or {}, requested_model)
    if loaded:
        evidence["lemonade_model"] = loaded
        evidence["model_match"] = {"mode": match_mode, "requested": requested_model, "selected": loaded.get("model_name")}
        backend_url = loaded.get("backend_url")
        pid = loaded.get("pid")
        if backend_url:
            root = endpoint_root(str(backend_url))
            for name, path in (("llama_props", "/props"), ("llama_slots", "/slots"), ("llama_metrics", "/metrics")):
                try:
                    if name == "llama_metrics":
                        request = urllib.request.Request(root + path, headers={"Accept": "text/plain"})
                        with urllib.request.urlopen(request, timeout=5) as response:
                            value: Any = response.read().decode("utf-8", "replace")
                    else:
                        value = get_json(root + path, 5)
                    evidence[name] = value
                    evidence["sources"].append({"name": name, "url": root + path, "confidence": "authoritative_runtime"})
                except Exception as exc:
                    evidence[name + "_error"] = repr(exc)
        if pid:
            try:
                pid_int = int(pid)
                cmd = _read_proc_cmdline(pid_int)
                exe = _proc_exe(pid_int)
                ports = _pid_listening_ports(pid_int)
                expected_port = urllib.parse.urlsplit(str(backend_url or "")).port
                evidence["process"] = {
                    "pid": pid_int,
                    "argv": cmd,
                    "executable": exe,
                    "start_ticks": _proc_start_ticks(pid_int),
                    "listening_ports": sorted(ports),
                    "backend_port_verified": expected_port is None or expected_port in ports or not ports,
                }
                evidence["sources"].append({"name": "proc_cmdline", "pid": pid_int, "confidence": "authoritative_process" if evidence["process"]["backend_port_verified"] else "process_unverified_port"})
            except (TypeError, ValueError):
                evidence["process_error"] = "invalid pid from health response"
    else:
        evidence["lemonade_model_error"] = "loaded model could not be matched; no process guessed"

    lemonade_base = str(config.get("lemonade_url") or "http://127.0.0.1:13305").rstrip("/")
    for path in ("/api/v1/system-info", "/v1/system-info"):
        try:
            evidence["system_info"] = get_json(lemonade_base + path, 5)
            evidence["sources"].append({"name": "lemonade_system_info", "url": lemonade_base + path, "confidence": "authoritative_host"})
            break
        except Exception:
            continue
    evidence["host"] = {
        "platform": os.uname().sysname + " " + os.uname().release if hasattr(os, "uname") else os.name,
        "python": os.sys.version.split()[0],
    }
    swap_process = _find_swap_process()
    if swap_process:
        evidence["llama_swap_process"] = swap_process
        evidence["sources"].append({"name": "llama_swap_process", "pid": swap_process["pid"], "confidence": "authoritative_supervisor"})
    emit("snapshot", {"requested_model": requested_model, "evidence_sources": evidence["sources"]})
    return evidence


def binary_version(binary: str) -> str | None:
    try:
        completed = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=3, check=False)
        output = (completed.stdout or completed.stderr).strip()
        return output[:500] if output else None
    except (OSError, subprocess.SubprocessError):
        return None
