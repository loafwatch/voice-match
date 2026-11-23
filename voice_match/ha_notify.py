"""Tell Home Assistant who spoke, via a Home Assistant webhook.

No Home Assistant token is needed: Home Assistant exposes a webhook (trigger
`webhook`, local only) and a trigger-based template sensor stores the
speaker. The webhook URL is install-specific and comes from VM_HA_WEBHOOK_URL.
"""

from __future__ import annotations

import asyncio
import json
import logging
import urllib.request
from typing import Optional

_LOGGER = logging.getLogger(__name__)


def _post(url: str, payload: dict, timeout: float) -> int:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status


async def send_speaker(
    url: Optional[str], speaker: str, score: Optional[float], timeout: float = 2.0
) -> None:
    """Post the speaker; failures are logged, never raised."""
    if not url:
        return
    payload = {"speaker": speaker, "score": None if score is None else round(score, 3)}
    try:
        await asyncio.to_thread(_post, url, payload, timeout)
    except Exception as err:  # noqa: BLE001 - HA being down must not break voice
        _LOGGER.warning("Could not update Home Assistant speaker sensor: %s", type(err).__name__)


async def send_status(url: Optional[str], payload: dict, timeout: float = 2.0) -> None:
    """Post enrolment status (state, name, done, total, message, voices)."""
    if not url:
        return
    try:
        await asyncio.to_thread(_post, url, payload, timeout)
    except Exception as err:  # noqa: BLE001
        _LOGGER.warning("Could not update Home Assistant enrolment sensor: %s", type(err).__name__)
