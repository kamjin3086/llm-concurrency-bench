"""Small, dependency-free authentication and redaction helpers."""
from __future__ import annotations

import base64
import getpass
import hashlib
import hmac
import json
import os
import re
import secrets
from pathlib import Path
from typing import Any


def hash_password(password: str) -> str:
    if len(password) < 10:
        raise ValueError("password must be at least 10 characters")
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=2**14, r=8, p=1)
    return "scrypt$16384$8$1$" + base64.urlsafe_b64encode(salt).decode() + "$" + base64.urlsafe_b64encode(digest).decode()


def verify_password(password: str, encoded: str) -> bool:
    try:
        scheme, n, r, p, salt_b64, digest_b64 = encoded.split("$", 5)
        if scheme != "scrypt":
            return False
        salt = base64.urlsafe_b64decode(salt_b64.encode())
        expected = base64.urlsafe_b64decode(digest_b64.encode())
        actual = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=int(n), r=int(r), p=int(p))
        return hmac.compare_digest(actual, expected)
    except (ValueError, TypeError):
        return False


def new_token() -> str:
    return secrets.token_urlsafe(32)


def _redact_string(value: str) -> str:
    home = str(Path.home())
    replacements = [
        (home, "~"),
        (f"/home/{getpass.getuser()}", "~"),
        (f"/Users/{getpass.getuser()}", "~"),
    ]
    for source, target in replacements:
        if source and source != target:
            value = value.replace(source, target)
    # Do not let a share image expose common local identity or credentials.
    value = re.sub(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+", r"\1[redacted]", value)
    value = re.sub(r"(?i)(api[_-]?key|token|secret|password)(\s*[=:]\s*)[^,\s}\]]+", r"\1\2[redacted]", value)
    value = re.sub(r"(?<![\w.])(?:127\.0\.0\.1|0\.0\.0\.0|localhost)(?=[:/\s]|$)", "local", value, flags=re.I)
    return value


def redact(value: Any, *, remove_prompts: bool = False) -> Any:
    """Recursively make a JSON-compatible object safe to share."""
    if isinstance(value, str):
        return "[prompt hidden]" if remove_prompts else _redact_string(value)
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            key_l = str(key).lower()
            if key_l in {"password", "api_key", "apikey", "token", "secret", "authorization", "cookie"}:
                result[str(key)] = "[redacted]"
            elif remove_prompts and key_l in {"prompt", "system_prompt", "messages", "preload_prompt"}:
                result[str(key)] = "[prompt hidden]"
            else:
                result[str(key)] = redact(item, remove_prompts=remove_prompts)
        return result
    if isinstance(value, list):
        return [redact(item, remove_prompts=remove_prompts) for item in value]
    if isinstance(value, tuple):
        return [redact(item, remove_prompts=remove_prompts) for item in value]
    return value


def json_safe(value: Any, *, remove_prompts: bool = False) -> str:
    return json.dumps(redact(value, remove_prompts=remove_prompts), ensure_ascii=False, sort_keys=True)
