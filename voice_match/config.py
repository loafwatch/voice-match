"""Settings from command-line flags, falling back to environment variables.

Nothing install-specific is hard-coded: the upstream address and any tokens
come from the container's environment / local config only.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field

from .trim import TrimSettings


@dataclass(frozen=True)
class Settings:
    uri: str
    upstream_uri: str
    upstream_timeout: float
    describe_timeout: float
    log_transcripts: bool
    debug: bool
    trim: TrimSettings = field(default_factory=TrimSettings)
    debug_audio_dir: str | None = None
    debug_audio_keep: int = 20


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
    parser.add_argument(
        "--trim",
        choices=["off", "edges", "first"],
        default=os.environ.get("VM_TRIM", "first"),
        help="off | edges (cut silence at both ends) | first (keep only the first burst of speech) (env VM_TRIM)",
    )
    parser.add_argument("--trim-gap-ms", type=float, default=float(os.environ.get("VM_TRIM_GAP_MS", "800")))
    parser.add_argument("--trim-pad-before-ms", type=float, default=float(os.environ.get("VM_TRIM_PAD_BEFORE_MS", "200")))
    parser.add_argument("--trim-pad-after-ms", type=float, default=float(os.environ.get("VM_TRIM_PAD_AFTER_MS", "300")))
    parser.add_argument("--vad-threshold", type=float, default=float(os.environ.get("VM_VAD_THRESHOLD", "0.5")))
    parser.add_argument(
        "--debug-audio-dir",
        default=os.environ.get("VM_DEBUG_AUDIO_DIR") or None,
        help="If set, save each command's audio before/after trimming here (env VM_DEBUG_AUDIO_DIR)",
    )
    parser.add_argument("--debug-audio-keep", type=int, default=int(os.environ.get("VM_DEBUG_AUDIO_KEEP", "20")))
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
        trim=TrimSettings(
            mode=args.trim,
            threshold=args.vad_threshold,
            gap_ms=args.trim_gap_ms,
            pad_before_ms=args.trim_pad_before_ms,
            pad_after_ms=args.trim_pad_after_ms,
        ),
        debug_audio_dir=args.debug_audio_dir,
        debug_audio_keep=args.debug_audio_keep,
    )
