"""Entry point: python -m voice_match"""

from __future__ import annotations

import asyncio
import logging
from functools import partial

from wyoming.server import AsyncServer

from .config import parse_settings
from .handler import ProxyEventHandler, fetch_upstream_info

_LOGGER = logging.getLogger("voice_match")


async def main() -> None:
    settings = parse_settings()
    logging.basicConfig(
        level=logging.DEBUG if settings.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    vad = None
    if settings.trim.mode != "off":
        from .vad import SileroVad

        vad = SileroVad()
        _LOGGER.info("Trimming: %s (gap %.0f ms)", settings.trim.mode, settings.trim.gap_ms)
    else:
        _LOGGER.info("Trimming: off")
    if settings.debug_audio_dir:
        _LOGGER.warning("Saving debug audio to %s (last %d commands)",
                        settings.debug_audio_dir, settings.debug_audio_keep)

    speaker = None
    if settings.data_dir:
        if vad is None:
            _LOGGER.warning("Speaker recognition needs trimming on; ignoring VM_DATA_DIR")
        else:
            from .handler import SpeakerContext
            from .speaker import SpeakerModel, Voiceprints

            from functools import partial as _partial

            from .control import ControlServer
            from .enrolment import Enrolment
            from .ha_notify import send_status

            model = SpeakerModel()
            voiceprints = Voiceprints(model, settings.data_dir)
            voiceprints.build()
            enrolment = Enrolment(voiceprints, _partial(send_status, settings.ha_enrol_webhook_url))
            speaker = SpeakerContext(model, voiceprints.prints, voiceprints, enrolment)
            _LOGGER.info("Speaker recognition: on (threshold %.2f)", settings.speaker.threshold)
            await send_status(settings.ha_enrol_webhook_url, enrolment.status())
            if settings.control_token:
                await ControlServer(enrolment, settings.control_token, port=settings.control_port).start()
            else:
                _LOGGER.info("Enrolment control API off (set VM_CONTROL_TOKEN to enable)")

    info_cache: dict = {}
    try:
        info_cache["info"] = await fetch_upstream_info(settings)
        _LOGGER.info("Upstream reachable: %s", [p.name for p in info_cache["info"].asr])
    except Exception as err:  # noqa: BLE001 - keep starting; HA retries describe
        _LOGGER.warning("Upstream not reachable yet (%s); will retry on describe", err)

    server = AsyncServer.from_uri(settings.uri)
    _LOGGER.info("Listening on %s", settings.uri)
    await server.run(partial(ProxyEventHandler, settings, info_cache, vad, speaker=speaker))


def run() -> None:
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    run()
