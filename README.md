# voice-match

A [Wyoming](https://github.com/OHF-Voice/wyoming) speech-to-text proxy for Home Assistant voice assistants. It sits between Home Assistant and your existing speech-to-text server (for example faster-whisper) and, as it grows, will:

1. **Trim** background speech (TV, other people) before and after the spoken command, so speech-to-text only hears the command.
2. **Recognise who is speaking** by comparing the voice with enrolled voiceprints, and publish the result to Home Assistant.
3. *(Later, optional)* isolate the speaker from voices talking over them.

It runs on the CPU and is designed to add well under half a second per command.

## Status

**Milestone 1 – pass-through.** Audio goes to the upstream server unchanged; the log line for each command shows the added latency (`proxy_overhead`). Trimming and recognition come next.

## How it works

```
Satellite → Home Assistant Assist pipeline → voice-match (this) → faster-whisper
                                                   │
                                                   └─ (soon) sensor with who spoke
```

Home Assistant talks to voice-match exactly as it would to any Wyoming speech-to-text server. voice-match buffers the utterance, forwards it upstream and relays the transcript back.

## Run it

```sh
docker run -d --name voice-match -p 10310:10310 \
  -e VM_UPSTREAM_URI=tcp://faster-whisper:10300 \
  ghcr.io/loafwatch/voice-match:latest
```

Then in Home Assistant: **Settings → Devices & services → Add integration → Wyoming Protocol**, host = the machine running voice-match, port `10310`. Select **voice-match** as the speech-to-text engine in your Assist pipeline.

| Variable | Default | Meaning |
|---|---|---|
| `VM_UPSTREAM_URI` | *(required)* | Upstream Wyoming STT server |
| `VM_URI` | `tcp://0.0.0.0:10310` | Where voice-match listens |
| `VM_UPSTREAM_TIMEOUT` | `30` | Seconds to wait for a transcript |
| `VM_LOG_TRANSCRIPTS` | `false` | Log transcript text (off by default for privacy) |
| `VM_DEBUG` | `false` | Debug logging |

## Development

```sh
pip install -e ".[dev]"
pytest -q
git config core.hooksPath .githooks   # privacy scan on commit and push
```

## Privacy

Voiceprints and recordings stay on your server and are never part of this repository. See `CLAUDE.md` for the repository's privacy rules.

Speaker recognition is good, not perfect (colds, shouting, similar voices). Never use it as the only protection for anything security-related.

## Licence

MIT
