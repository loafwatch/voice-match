#!/bin/sh
# Start as root only long enough to make the data folder writable, then run
# voice-match as an unprivileged user (Unraid convention: nobody:users = 99:100).
set -e
PUID="${PUID:-99}"
PGID="${PGID:-100}"
if [ "$(id -u)" = "0" ]; then
    if [ -n "$VM_DATA_DIR" ]; then
        mkdir -p "$VM_DATA_DIR"
        chown -R "$PUID:$PGID" "$VM_DATA_DIR"
        chmod 700 "$VM_DATA_DIR"
    fi
    exec setpriv --reuid="$PUID" --regid="$PGID" --clear-groups voice-match "$@"
fi
exec voice-match "$@"
