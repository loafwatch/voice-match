FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY scripts/fetch_models.py ./scripts/
COPY voice_match ./voice_match
# Silero VAD model (MIT), pinned and checksum-verified
# Speaker model: WeSpeaker ResNet34-LM (CC BY 4.0), pinned and checksum-verified
RUN python scripts/fetch_models.py \
    && pip install --no-cache-dir .
COPY docker-entrypoint.sh /usr/local/bin/

ENV HOME=/tmp
EXPOSE 10310

# Required: VM_UPSTREAM_URI, e.g. tcp://faster-whisper:10300
# Runs as PUID:PGID (default 99:100) after preparing VM_DATA_DIR.
ENTRYPOINT ["docker-entrypoint.sh"]
