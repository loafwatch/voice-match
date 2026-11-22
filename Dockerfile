FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY voice_match ./voice_match
RUN pip install --no-cache-dir . \
    && useradd --system --uid 10310 voicematch

USER voicematch
EXPOSE 10310

# Required: VM_UPSTREAM_URI, e.g. tcp://faster-whisper:10300
ENTRYPOINT ["voice-match"]
