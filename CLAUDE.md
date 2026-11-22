# voice-match

A Wyoming speech-to-text proxy for Home Assistant (Python, `wyoming` library). Sits in front of an existing faster-whisper Wyoming server; trims non-command speech, recognises the speaker against enrolled voiceprints, and reports the speaker to Home Assistant.

## Privacy rules (public repository)

This repository is public. It must never contain personal data about the household it runs in.

- No real names (people, pets, rooms named after people), addresses, towns, hostnames, domains, IP addresses, email addresses or Home Assistant entity IDs from a real install.
- No voiceprints, embeddings, recordings or transcripts from a real install – not in code, tests, fixtures, docs or issues. Tests use synthetic audio only.
- Anything install-specific (upstream address, Home Assistant URL and token, speaker names, thresholds) comes from environment variables or a local, git-ignored config file.
- Examples, docs and tests use made-up values only: "Speaker A", `sensor.example_speaker`, `example.com`.
- Commit messages and PR text follow the same rules.
- `scripts/privacy_scan.py` runs in the pre-commit, commit-msg and pre-push hooks (`git config core.hooksPath .githooks`) and in CI. Never bypass it (no `--no-verify`). If it flags something, remove the data; do not weaken the scan.
- The private words list lives only in `.private-terms` (git-ignored, local to the working copy). It is never committed, never stored on GitHub and never printed.

## Design constraints

- Runs on the CPU. Do not use the GPU – it is shared with a local LLM and filling it slows the voice assistant.
- Must add well under 0.5 s per command. Log timing for every request.
- Transcript text is not logged unless `VM_LOG_TRANSCRIPTS` is on.
- Only permissively licensed dependencies and models (MIT, Apache-2.0, BSD). Do not copy code from non-commercial / no-derivatives licensed projects.
