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

Enrolment is a guided session started from Home Assistant (for example a button on a dashboard):

1. Set `VM_CONTROL_TOKEN` (a long random string) and publish port `10311`. Home Assistant calls the control API with a `rest_command`, sending the token in the `X-Voice-Match-Token` header:
   - `POST /enrol/start` `{"name": "speaker_a", "total": 5, "mode": "replace"}` (`mode: "add"` tops up an existing voice instead of replacing it)
   - `POST /enrol/cancel`
   - `POST /voices/delete` `{"name": "speaker_a"}`
   - `GET /status`
2. The person then says the wake word and reads a sentence, `total` times. Each sample's speaker embedding (256 numbers) is saved to `VM_DATA_DIR/embeddings/<name>/`; **no audio is ever written to disk**, and an embedding cannot be turned back into speech. Instead of transcribing, voice-match answers with a status transcript for a Home Assistant sentence automation to reply to:
   - `voice enrolment sample <done> of <total>`
   - `voice enrolment complete <name>`
   - `voice enrolment retry` (less than 1.5 s of speech)
   - `voice enrolment other <name>` (sounded like someone already enrolled – rejected)
3. After the last sample the voiceprint is rebuilt and used straight away – no restart. Sessions time out after 5 minutes without a sample; cancelling discards that session's samples.

Progress is posted to `VM_HA_ENROL_WEBHOOK_URL` as `{"state", "name", "done", "total", "message", "voices": [{"name", "recordings", "built"}]}` – handy for a trigger-based template sensor that a dashboard can show.

Enrolment data from older versions (`enrol/<name>/*.wav`) is converted to embeddings at start-up and the `.wav` files deleted; nothing else in the folder is touched.

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
