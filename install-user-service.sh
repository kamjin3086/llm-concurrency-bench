#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf '%s\n' \
    'Install BenchRoom as a user systemd service.' \
    '' \
    'Usage: ./install-user-service.sh [options]' \
    '' \
    'Options:' \
    '  --host HOST       Listen address (default: 0.0.0.0)' \
    '  --port PORT       Listen port (default: 8790)' \
    '  --db PATH         SQLite path (default: ~/.local/share/llm-concurrency-bench/bench.db)' \
    '  --no-linger       Do not request start-at-boot user lingering' \
    '  --uninstall       Stop, disable, and remove the installed unit' \
    '  -h, --help        Show this help'
}

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
user_home="$(getent passwd "$(id -u)" | cut -d: -f6)"
user_home="${user_home:-$(pwd -P)}"
config_root="${XDG_CONFIG_HOME:-${user_home}/.config}"
data_root="${XDG_DATA_HOME:-${user_home}/.local/share}"
unit_dir="${config_root}/systemd/user"
unit_file="${unit_dir}/llm-concurrency-bench.service"
template="${project_dir}/systemd/llm-concurrency-bench.service.template"
host="0.0.0.0"
port="8790"
db_path="${data_root}/llm-concurrency-bench/bench.db"
no_linger=0
uninstall=0

while (($#)); do
  case "$1" in
    --host) [[ $# -ge 2 ]] || { printf '%s\n' '--host needs a value' >&2; exit 2; }; host="$2"; shift 2 ;;
    --port) [[ $# -ge 2 ]] || { printf '%s\n' '--port needs a value' >&2; exit 2; }; port="$2"; shift 2 ;;
    --db) [[ $# -ge 2 ]] || { printf '%s\n' '--db needs a value' >&2; exit 2; }; db_path="$2"; shift 2 ;;
    --no-linger) no_linger=1; shift ;;
    --uninstall) uninstall=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown option: %s\n\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

command -v systemctl >/dev/null 2>&1 || { printf '%s\n' 'systemctl is required.' >&2; exit 1; }

if ((uninstall)); then
  systemctl --user disable --now llm-concurrency-bench.service >/dev/null 2>&1 || true
  if [[ -f "$unit_file" ]]; then
    rm -f -- "$unit_file"
    systemctl --user daemon-reload
  fi
  printf 'Removed user service: %s\n' "$unit_file"
  exit 0
fi

[[ "$port" =~ ^[0-9]+$ ]] && ((port >= 1 && port <= 65535)) || { printf '%s\n' 'Port must be an integer from 1 to 65535.' >&2; exit 2; }
[[ "$host" != *$'\n'* && "$host" != *$'\r'* && "$host" != *'"'* ]] || { printf '%s\n' 'Host contains invalid characters.' >&2; exit 2; }
[[ -f "$template" ]] || { printf 'Missing service template: %s\n' "$template" >&2; exit 1; }
python_bin="$(command -v python3 || command -v python || true)"
[[ -n "$python_bin" ]] || { printf '%s\n' 'python3 (or python) is required.' >&2; exit 1; }
db_path="$(python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).expanduser().resolve())' "$db_path")"

install -d -m 0755 -- "$unit_dir"
tmp_unit="$(mktemp "${unit_file}.XXXXXX")"
trap 'rm -f -- "$tmp_unit"' EXIT

python3 - "$template" "$tmp_unit" "$project_dir" "$python_bin" "$host" "$port" "$db_path" <<'PY'
import pathlib
import sys

template, output, project, python_bin, host, port, db = sys.argv[1:]

def unit_quote(value: str) -> str:
    # systemd supports quoted values; escape its two special characters here.
    return value.replace('\\', '\\\\').replace('"', '\\"').replace('%', '%%').replace('\n', '').replace('\r', '')

text = pathlib.Path(template).read_text(encoding='utf-8')
for key, value in {
    '__PROJECT_DIR__': project,
    '__PYTHON_BIN__': python_bin,
    '__HOST__': host,
    '__PORT__': port,
    '__DB_PATH__': db,
}.items():
    text = text.replace(key, unit_quote(value))
pathlib.Path(output).write_text(text, encoding='utf-8')
PY

install -m 0644 -- "$tmp_unit" "$unit_file"
systemctl --user daemon-reload
systemctl --user enable --now llm-concurrency-bench.service

if ((no_linger == 0)) && command -v loginctl >/dev/null 2>&1; then
  if loginctl enable-linger "$(id -un)" >/dev/null 2>&1; then
    printf '%s\n' 'User lingering enabled: service can start at boot without an interactive login.'
  else
    printf '%s\n' 'Warning: could not enable user lingering; service will start when your user session starts.' >&2
  fi
fi

printf '\nInstalled and started: %s\n' "$unit_file"
printf 'URL: http://%s:%s\n' "$host" "$port"
printf 'Status: systemctl --user status llm-concurrency-bench.service\n'
