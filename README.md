# voice-match

A [Wyoming](https://github.com/OHF-Voice/wyoming) speech-to-text proxy for Home Assistant voice assistants. It sits between Home Assistant and your existing speech-to-text server (for example faster-whisper) and, as it grows, will:

1. **Trim** background speech (TV, other people) before and after the spoken command, so speech-to-text only hears the command.
2. **Recognise who is speaking** by comparing the voice with enrolled voiceprints, and publish the result to Home Assistant.
3. *(Later, optional)* isolate the speaker from voices talking over them.

It runs on the CPU and is designed to add well under half a second per command.

## Status

**Milestone 1 – trimming.** [Silero VAD](https://github.com/snakers4/silero-vad) (MIT, run with onnxruntime on the CPU) scores the audio as it streams in. At the end of the command, voice-match keeps only the first burst of speech (plus a little padding) and sends that upstream, so a TV or conversation carrying on after the command is not transcribed. If no speech is found, the whole recording is sent. Speaker recognition comes next.

Each command logs one line: recording length, how much was sent, what was trimmed, time spent in the VAD and the added latency (`proxy_overhead`).

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
| `VM_TRIM` | `first` | `first` = keep only the first burst of speech; `edges` = only cut silence before and after; `off` |
| `VM_TRIM_GAP_MS` | `800` | In `first` mode, a silence longer than this ends the command |
| `VM_TRIM_PAD_BEFORE_MS` / `VM_TRIM_PAD_AFTER_MS` | `200` / `300` | Audio kept either side of the speech |
| `VM_VAD_THRESHOLD` | `0.5` | Speech probability needed to count as speech |
| `VM_DEBUG_AUDIO_DIR` | *(unset)* | If set, saves each command's audio before/after trimming (for tuning; stays on your server) |
| `VM_DEBUG_AUDIO_KEEP` | `20` | How many commands of debug audio to keep |
| `VM_LOG_TRANSCRIPTS` | `false` | Log transcript text (off by default for privacy) |
| `VM_DEBUG` | `false` | Debug logging |

## Development

```sh
python scripts/fetch_models.py   # Silero VAD model, pinned + checksum-checked
pip install -e ".[dev]"
pytest -q
git config core.hooksPath .githooks   # privacy scan on commit and push
```

## Privacy

Voiceprints and recordings stay on your server and are never part of this repository. See `CLAUDE.md` for the repository's privacy rules.

Speaker recognition is good, not perfect (colds, shouting, similar voices). Never use it as the only protection for anything security-related.

## Licence

MIT. The Silero VAD model is MIT-licensed by the Silero Team; it is downloaded at build time, not stored in this repository.
