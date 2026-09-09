#!/bin/bash
# jarvis-waybar.sh
# Make it executable: chmod +x jarvis-waybar.sh

STATE_FILE="/tmp/jarvis_state"

# Handle click event (toggle Jarvis on/off)
if [ "$1" == "toggle" ]; then
    if pgrep -f "agent_v2.py" > /dev/null; then
        pkill -f "agent_v2.py"
        echo "offline" > "$STATE_FILE"
    else
        # Launch Jarvis in the background (no terminal)
        cd ~/Desktop/LLM && .venv/bin/python agent_v2.py > /tmp/jarvis_logs.txt 2>&1 &
    fi
    exit 0
fi

# Output JSON for Waybar
state=$(cat "$STATE_FILE" 2>/dev/null || echo "offline")

case $state in
    "sleeping"|"idle") icon="💤" ; class="sleeping" ;;
    "listening") icon="🎤" ; class="listening" ;;
    "thinking")  icon="🧠" ; class="thinking" ;;
    "speaking")  icon="🔊" ; class="speaking" ;;
    "error")     icon="⚠️" ; class="error" ;;
    *)           icon="🤖" ; class="offline" ;;
esac

echo "{\"text\": \"$icon\", \"class\": \"$class\", \"tooltip\": \"J.A.R.V.I.S: $state\"}"
