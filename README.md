# BenchRoom

BenchRoom is a local-first browser workbench for measuring LLM concurrency on the machine that actually serves the model. It is designed around a llama-swap → Lemonade → llama.cpp setup, while keeping a basic OpenAI-compatible fallback.

## Why this exists

The old workflow required editing JSON, running a terminal command, and interpreting a CSV that was overwritten on the next run. BenchRoom keeps the benchmark engine, stores every run in SQLite, streams progress to the browser, and turns finished runs into a comparison/report view that is easy to screenshot.

## Start it

Python 3.10+ is enough; the web server has no runtime dependencies.

```bash
python3 -m backend.server --host 0.0.0.0 --port 8790
```

Open `http://<this-machine>:8790` from the machine or a trusted LAN device. On the first visit, set a password of at least 10 characters. The first start creates the local database at `~/.local/share/llm-concurrency-bench/bench.db` and seeds a llama-swap endpoint plus three presets.

For a one-time password change:

```bash
python3 benchroom.py set-password
```

To install and start a user service in one step:

```bash
./install-user-service.sh
```

The script installs the unit for the current checkout, runs `daemon-reload`, enables and starts it, and requests user lingering so it can start at boot without an interactive login. Use `--no-linger` if that policy is managed elsewhere, or `--uninstall` to remove the unit. `--host`, `--port`, and `--db` override the defaults.

The existing headless command remains available:

```bash
./llm_concurrency_bench.py bench.example.json
```

## Daily workflow

1. Refresh the model list from the configured endpoint. IDs are grouped by a separator-aware longest common prefix, so new naming conventions and variants do not need a hardcoded suffix list.
2. Check one or more model variants, or select an entire group from its parent row.
3. Pick `快速检查`, `标准测试`, or `稳定性测试` and adjust the few fields that matter.
4. Start the run. Models execute in order and one benchmark owns the machine at a time.
5. Open the generated report, review the peak and balanced (“sweet spot”) concurrency, then download a redacted PNG.

The full request, each response timing, environment details, and runtime evidence stay in the local database. The share view hides prompts, user names, absolute paths, local addresses, credentials, and raw error details.

## Runtime parameter evidence

When the endpoint is Lemonade-backed, a run captures the following after a real preload completion:

1. Lemonade health data: model, checkpoint, recipe, device, backend URL, PID, recipe options, and (when exposed) its expanded launch command.
2. The verified backend process command line from `/proc/<pid>/cmdline`, with executable, start time and listening-port checks.
3. Direct llama.cpp `/props`, `/slots`, and `/metrics` responses.
4. Host information from Lemonade system-info.

Each field is labelled with its source and confidence. Logs are retained only as optional diagnostics; they are not treated as the source of truth. This matters for options such as `--fit`, where a configured value and the server’s effective context can differ.

## Project layout

- `backend/engine.py` — streaming runner, metrics, runtime evidence collector
- `backend/store.py` — SQLite schema and history persistence
- `backend/server.py` — authenticated web API, job queue and static server
- `web/` — responsive dashboard, live progress, history, compare and share view
- `llm_concurrency_bench.py` — original zero-dependency CLI retained for compatibility

## Privacy and publishing

The public repository intentionally contains no database, results, credentials, host paths, or generated screenshots. `.gitignore` excludes local history, `.env` files, keys, logs, caches, and benchmark output. Keep the service on a trusted network unless you put it behind HTTPS and an additional network control; it can start expensive model generations.
