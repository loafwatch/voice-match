# voice-match

A [Wyoming](https://github.com/OHF-Voice/wyoming) speech-to-text proxy for Home Assistant voice assistants. It sits between Home Assistant and your existing speech-to-text server (for example faster-whisper) and, as it grows, will:

1. **Trim** background speech (TV, other people) before and after the spoken command, so speech-to-text only hears the command.
2. **Recognise who is speaking** by comparing the voice with enrolled voiceprints, and publish the result to Home Assistant.
3. *(Later, optional)* isolate the speaker from voices talking over them.

It runs on the CPU and is designed to add well under half a second per command.

## Status

**Milestone 1 – trimming.** [Silero VAD](https://github.com/snakers4/silero-vad) (MIT, run with onnxruntime on the CPU) scores the audio as it streams in. At the end of the command, voice-match keeps only the first burst of speech (plus a little padding) and sends that upstream, so a TV or conversation carrying on after the command is not transcribed. If no speech is found, the whole recording is sent.

**Early end:** in `first` mode, as soon as the first burst of speech is followed by more than `VM_TRIM_GAP_MS` of non-speech, voice-match transcribes and answers straight away. Home Assistant stops recording when the transcript arrives (Home Assistant 2026.11 and later reads the reply while still sending audio), so a TV talking in the background no longer keeps the assistant listening.

**Speaker recognition (optional):** enrol household voices and voice-match scores a 1-second window every 250 ms against each voiceprint, using the [WeSpeaker](https://github.com/wenet-e2e/wespeaker) ResNet34-LM model (CC BY 4.0) on the CPU. It then keeps only the part of the recording where a household voice was speaking and ends the command once that voice stops – even if a TV talks the whole time. The speaker's name and match score are logged. If no enrolled voice is found, it falls back to the trimming above.

### "Last speaker" sensor in Home Assistant

voice-match can post who gave each command to a Home Assistant webhook – no Home Assistant token needed. Add a trigger-based template sensor (e.g. in a package), using a long random webhook ID:

```yaml
template:
  - triggers:
      - trigger: webhook
        webhook_id: "<long random id>"
        local_only: true
        allowed_methods: [POST]
    sensor:
      - name: Voice match last speaker
        unique_id: voice_match_last_speaker
        icon: mdi:account-voice
        state: "{{ trigger.json.speaker }}"
        attributes:
          score: "{{ trigger.json.score }}"
          spoken_at: "{{ now().isoformat() }}"
```

Then set `VM_HA_WEBHOOK_URL` to `http://<home-assistant>:8123/api/webhook/<long random id>`. The sensor is updated while the command is being transcribed, so it is current by the time the conversation agent and any automations run. The state is the enrolled name, or `unknown`.

### Enrolling voices

1. Set `VM_DATA_DIR` (and mount a folder there) and `VM_ENROLL=<name>` (letters, digits, `_` or `-`), then restart.
2. That person gives 10–15 ordinary commands at the satellite, at normal distance, with the room quiet. Each command's speech is saved to `VM_DATA_DIR/enrol/<name>/`.
3. Remove `VM_ENROLL` and restart. Voiceprints are built at start-up into `VM_DATA_DIR/voiceprints/` (with a `.json` of quality stats). Repeat for each person.

Delete a person's `enrol/<name>` folder and `voiceprints/<name>.*` files to remove them.

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
| `VM_EARLY_END` | `true` | Answer as soon as the command ends instead of waiting for Home Assistant to stop recording |
| `VM_VAD_THRESHOLD` | `0.5` | Speech probability needed to count as speech |
| `VM_DATA_DIR` | *(unset)* | Folder for enrolment recordings and voiceprints; enables speaker recognition |
| `VM_ENROLL` | *(unset)* | Enrolment mode: save each command as a recording of this person |
| `VM_SPEAKER_THRESHOLD` | `0.45` | Similarity (0–1) a window needs to count as an enrolled voice |
| `VM_SPEAKER_GAP_MS` | `1000` | Enrolled voice not heard for this long = command over |
| `VM_HA_WEBHOOK_URL` | *(unset)* | Home Assistant webhook that receives `{"speaker", "score"}` after each command |
| `PUID` / `PGID` | `99` / `100` | User the container runs as (owns `VM_DATA_DIR`) |
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

MIT. Models are downloaded at build time, not stored in this repository:
- Silero VAD – MIT, © Silero Team.
- WeSpeaker ResNet34-LM speaker embedding model – trained on VoxCeleb by the WeSpeaker project, CC BY 4.0; ONNX export from sherpa-onnx. Used unmodified.
