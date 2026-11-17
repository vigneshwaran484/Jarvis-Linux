"""
bench_tools.py — measure how well an Ollama model drives Jarvis's tools.

For every test utterance it runs the real system prompt through the model and
checks that the FIRST step picks the expected tool (and argument, where one is
given). Nothing is executed: no apps open, no volume changes.

Run:  .venv/bin/python bench_tools.py llama3.2 qwen3:4b
      .venv/bin/python bench_tools.py            # just the configured LLM_MODEL
"""
import os, re, sys, tempfile, time

os.environ.setdefault("LOG_FILE", os.path.join(tempfile.mkdtemp(), "bench.log"))

import agent_v2  # noqa: E402

# (utterance, expected tool, substring the argument must contain or None)
CASES = [
    ("what's on my calendar tomorrow",           "get_calendar",    "tomorrow"),
    ("do I have anything on friday",             "get_calendar",    "fri"),
    ("what does my week look like",              "get_calendar",    "week"),
    ("when is my next exam",                     "get_calendar",    None),
    ("any events today",                         "get_calendar",    "today"),
    ("good morning",                             "daily_briefing",  None),
    ("brief me on my day",                       "daily_briefing",  None),
    ("what's the weather like in Mumbai",        "get_weather",     "mumbai"),
    ("is it going to be hot outside",            "get_weather",     None),
    ("what time is it",                          "get_datetime",    None),
    ("how much battery do I have left",          "get_battery",     None),
    ("how much RAM am I using",                  "get_system_stats", None),
    ("open spotify",                             "open_app",        "spotify"),
    ("close firefox",                            "close_app",       "firefox"),
    ("set the volume to 40 percent",             "set_volume",      "40"),
    ("make the screen a bit dimmer, like 30",    "set_brightness",  "30"),
    ("mute the sound",                           "mute_toggle",     None),
    ("skip this song",                           "media_next",      None),
    ("pause the music",                          "media_play_pause", None),
    ("take a screenshot",                        "take_screenshot", None),
    ("lock my computer",                         "lock_screen",     None),
    ("who wrote the Ramayana",                   "search_web",      "ramayana"),
    ("what is the capital of Australia",         "search_web",      "australia"),
    ("show me python tutorials in the browser",  "open_in_browser", "python"),
    ("go to workspace 3",                        "change_workspace", "3"),
    ("what windows do I have open",              "list_windows",    None),
    ("which apps are running right now",         "list_windows",    None),
    ("take me to my browser",                    "focus_window",    "browser"),
    ("switch to spotify",                        "focus_window",    "spotify"),
    ("jump to vs code",                          "focus_window",    "code"),
    ("move this window to workspace 5",          "move_to_workspace", "5"),
    ("send this to workspace two",               "move_to_workspace", None),
    ("what do you remember about my project",    "recall_memory",   None),
    ("start recording my screen",                "start_recording", None),
    ("read the file notes.txt",                  "read_file",       "notes"),
    ("what does this error on my screen mean",   "inspect_screen",  None),
    ("read what's in this window",               "inspect_screen",  None),
    ("can you explain the code I'm looking at",  "inspect_screen",  None),
    ("take a screenshot and save it",            "take_screenshot", None),
]


def first_step(utterance):
    messages = [{"role": "system", "content": agent_v2.SYSTEM_PROMPT},
                {"role": "user", "content": utterance}]
    t = time.time()
    try:
        out = agent_v2._ask_llm_raw(messages)
    except agent_v2.LLMError as e:
        return f"<error: {e}>", time.time() - t
    return out, time.time() - t


def parse(response):
    first = (response.strip().splitlines() or [""])[0]
    body = re.sub(r"^tool\s*:\s*", "", first, flags=re.I).strip()
    parts = body.split("|")
    return parts[0].strip().lower().replace(" ", "_"), "|".join(parts[1:]).lower()


def bench(model):
    agent_v2.LLM_MODEL = model
    first_step("hello")                      # load the model; not counted
    passed, times, misses = 0, [], []
    for utterance, tool, arg in CASES:
        out, dt = first_step(utterance)
        times.append(dt)
        name, args = parse(out)
        ok = name == tool and (arg is None or arg in args)
        passed += ok
        if not ok:
            misses.append((utterance, tool, out[:90].replace("\n", " / ")))
    times.sort()
    print(f"\n=== {model}: {passed}/{len(CASES)} correct, "
          f"median {times[len(times)//2]:.2f}s, slowest {times[-1]:.2f}s")
    for utterance, tool, out in misses:
        print(f"  ✗ {utterance!r}: wanted {tool}, got {out!r}")
    return passed


if __name__ == "__main__":
    for m in (sys.argv[1:] or [agent_v2.LLM_MODEL]):
        bench(m)
