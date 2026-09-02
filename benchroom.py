#!/usr/bin/env python3
"""BenchRoom launcher. The original llm_concurrency_bench.py remains available."""
from __future__ import annotations

import argparse
import getpass
import os
import sys

from backend.security import hash_password
from backend.server import DEFAULT_DB, App, main as serve_main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="benchroom", description="LLM concurrency benchmark workbench")
    sub = parser.add_subparsers(dest="command")
    serve = sub.add_parser("serve", help="start the web application")
    serve.add_argument("--host", default=os.environ.get("LLM_BENCH_HOST", "0.0.0.0"))
    serve.add_argument("--port", type=int, default=int(os.environ.get("LLM_BENCH_PORT", "8790")))
    serve.add_argument("--db", default=str(DEFAULT_DB))
    setup = sub.add_parser("set-password", help="set the web login password")
    setup.add_argument("--db", default=str(DEFAULT_DB))
    args = parser.parse_args(argv)
    if args.command == "serve":
        return serve_main(["--host", args.host, "--port", str(args.port), "--db", args.db])
    if args.command == "set-password":
        first = getpass.getpass("New BenchRoom password (10+ chars): ")
        second = getpass.getpass("Repeat password: ")
        if first != second:
            print("Passwords do not match", file=sys.stderr); return 2
        store = App(args.db).store
        store.set_setting("password_hash", hash_password(first))
        print(f"Password updated in {args.db}")
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
