"""Settings from command-line flags, falling back to environment variables.

Nothing install-specific is hard-coded: the upstream address and any tokens
come from the container's environment / local config only.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    uri: str
    upstream_uri: str
    upstream_timeout: float
    describe_timeout: float
    log_transcripts: bool
    debug: bool


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def parse_settings(argv: list[str] | None = None) -> Settings:
    parser = argparse.ArgumentParser(prog="voice-match")
    parser.add_argument(
        "--uri",
        default=os.environ.get("VM_URI", "tcp://0.0.0.0:10310"),
        help="Address this proxy listens on (env VM_URI)",
    )
    parser.add_argument(
        "--upstream-uri",
        default=os.environ.get("VM_UPSTREAM_URI"),
        help="Upstream Wyoming speech-to-text server, e.g. tcp://whisper:10300 (env VM_UPSTREAM_URI)",
    )
    parser.add_argument(
        "--upstream-timeout",
        type=float,
        default=float(os.environ.get("VM_UPSTREAM_TIMEOUT", "30")),
    )
    parser.add_argument(
        "--describe-timeout",
        type=float,
        default=float(os.environ.get("VM_DESCRIBE_TIMEOUT", "5")),
    )
    parser.add_argument(
        "--log-transcripts",
        action="store_true",
        default=_env_bool("VM_LOG_TRANSCRIPTS", False),
        help="Include transcript text in logs (off by default for privacy)",
    )
    parser.add_argument("--debug", action="store_true", default=_env_bool("VM_DEBUG", False))
    args = parser.parse_args(argv)

    if not args.upstream_uri:
        parser.error("--upstream-uri (or VM_UPSTREAM_URI) is required")

    return Settings(
        uri=args.uri,
        upstream_uri=args.upstream_uri,
        upstream_timeout=args.upstream_timeout,
        describe_timeout=args.describe_timeout,
        log_transcripts=args.log_transcripts,
        debug=args.debug,
    )
