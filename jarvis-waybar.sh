#!/bin/bash
# jarvis-waybar.sh
# Make it executable: chmod +x jarvis-waybar.sh

STATE_FILE="/tmp/jarvis_state"
JARVIS_DIR="$HOME/Desktop/LLM"

# Match ONLY the real interpreter process.
#
# This used to be `pgrep -f "agent_v2.py"`, an unanchored substring match against
# every process's full command line. Anything that merely MENTIONED the filename
# matched too -- an editor with the file open, a grep, a shell running a command
# containing the name. The toggle then believed Jarvis was already running, took
# the stop branch, and pkill'd that unrelated process instead of starting Jarvis.
JARVIS_PATTERN='^([^ ]*/)?python[0-9.]*[[:space:]]+([^ ]*/)?agent_v2\.py([[:space:]]|$)'

jarvis_pids() { pgrep -f "$JARVIS_PATTERN" 2>/dev/null; }

# Handle click event (toggle Jarvis on/off)
if [ "$1" == "toggle" ]; then
    pids=$(jarvis_pids)
    if [ -n "$pids" ]; then
        # shellcheck disable=SC2086
        kill $pids 2>/dev/null
        sleep 1
        still=$(jarvis_pids)
        # shellcheck disable=SC2086
        [ -n "$still" ] && kill -9 $still 2>/dev/null
        echo "offline" > "$STATE_FILE"
    else
        # Launch Jarvis in the background (no terminal).
        # reactor.py mirrors each state change to $STATE_FILE from here on; seed it
        # so a failed launch doesn't leave the previous run's state showing.
        echo "starting" > "$STATE_FILE"
        cd "$JARVIS_DIR" && .venv/bin/python agent_v2.py > /tmp/jarvis_logs.txt 2>&1 &
        disown
    fi
    exit 0
fi

# Report live process state rather than trusting a file that goes stale if the
# agent is killed from outside this script.
if [ -z "$(jarvis_pids)" ]; then
    state="offline"
else
    state=$(cat "$STATE_FILE" 2>/dev/null || echo "idle")
    [ "$state" = "offline" ] && state="idle"
fi

case $state in
    "sleeping"|"idle") icon="💤" ; class="sleeping" ;;
    "listening") icon="🎤" ; class="listening" ;;
    "thinking")  icon="🧠" ; class="thinking" ;;
    "speaking")  icon="🔊" ; class="speaking" ;;
    "starting")  icon="⏳" ; class="starting" ;;
    "error")     icon="⚠️" ; class="error" ;;
    *)           icon="🤖" ; class="offline" ;;
esac

echo "{\"text\": \"$icon\", \"class\": \"$class\", \"tooltip\": \"J.A.R.V.I.S: $state\"}"
