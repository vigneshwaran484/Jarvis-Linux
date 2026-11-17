"""
test_calendar.py — date-phrase parsing, clash detection and the spoken summary
of calendar_tool. No khal, no mic, no Ollama, no network needed.

Run:  .venv/bin/python test_calendar.py
"""
import sys
from datetime import date

import calendar_tool as ct
from calendar_tool import Event

ok = True
def check(cond, msg):
    global ok
    print(("✅ " if cond else "❌ ") + msg)
    ok &= bool(cond)

FRI = date(2026, 11, 13)   # a Friday

# 1. resolve_when
cases = [
    ("",                     (FRI, 1)),
    ("today",                (FRI, 1)),
    ("tomorrow",             (date(2026, 11, 14), 1)),
    ("day after tomorrow",   (date(2026, 11, 15), 1)),
    ("what's on tomorrow?",  (date(2026, 11, 14), 1)),
    ("monday",               (date(2026, 11, 16), 1)),
    ("friday",               (FRI, 1)),
    ("next friday",          (date(2026, 11, 20), 1)),
    ("this week",            (FRI, 3)),                       # Fri, Sat, Sun
    ("what does my week look like", (FRI, 3)),
    ("next week",            (date(2026, 11, 16), 7)),
    ("weekend",              (date(2026, 11, 14), 2)),
    ("2026-12-25",           (date(2026, 12, 25), 1)),
    ("20 november",          (date(2026, 11, 20), 1)),
    ("november 20th",        (date(2026, 11, 20), 1)),
    ("20th of nov",          (date(2026, 11, 20), 1)),
    ("5 january",            (date(2027, 1, 5), 1)),          # already past this year
    ("31 february",          (FRI, 1)),                       # impossible date -> today
    ("gibberish",            (FRI, 1)),
]
for text, want in cases:
    start, days, _ = ct.resolve_when(text, today=FRI)
    check((start, days) == want, f"resolve_when({text!r}) -> {start} x{days}")

check(ct.resolve_when("weekend", today=date(2026, 11, 15))[:2] == (date(2026, 11, 15), 1),
      "weekend asked on a Sunday is just that Sunday")

# 2. spoken times
check(ct._spoken_time("08:30") == "8:30 AM", "08:30 -> 8:30 AM")
check(ct._spoken_time("17:00") == "5 PM", "17:00 -> 5 PM")
check(ct._spoken_time("00:05") == "12:05 AM", "00:05 -> 12:05 AM")
check(ct._spoken_time("12:00") == "12 PM", "12:00 -> 12 PM")

# 3. conflicts
a = Event(FRI, "Team sync", "11:00", "12:00")
b = Event(FRI, "Lab review", "11:30", "13:00")
c = Event(FRI, "Gym", "12:00", "13:00")          # touches a, overlaps b
d = Event(FRI, "Practicals", all_day=True)
e = Event(date(2026, 11, 14), "Other day", "11:00", "12:00")
clashes = ct.find_conflicts([a, b, c, d, e])
names = {(x.title, y.title) for x, y in clashes}
check(("Team sync", "Lab review") in names, "overlapping events clash")
check(("Team sync", "Gym") not in names, "back-to-back events do not clash")
check(("Lab review", "Gym") in names, "second overlap found")
check(all("Practicals" not in pair for pair in names), "all-day events never clash")
check(all("Other day" not in pair for pair in names), "events on different days never clash")
check(Event(FRI, "Late", "23:00", "01:00").minutes() == (23 * 60, 24 * 60),
      "event running past midnight is clipped to the end of the day")

# 4. summary text
s = ct.summarize([d, a, b], "Today", today=FRI)
check(s.startswith("Today: 3 events."), "summary leads with the count")
check("all day: Practicals." in s and "11 AM to 12 PM: Team sync." in s, "summary lists events with spoken times")
check("CLASHES: 'Team sync' overlaps 'Lab review'." in s, "summary reports the clash")
check(ct.summarize([], "Tomorrow", today=FRI) == "Tomorrow: nothing on the calendar.", "empty day")
check(ct.summarize([a], "Today", today=FRI).startswith("Today: 1 event."), "singular for one event")
many = [Event(FRI, f"E{i}", all_day=True) for i in range(15)]
check("And 3 more." in ct.summarize(many, "Today", today=FRI), "long days are capped")
s = ct.summarize([a, e], "This week", today=FRI, multi_day=True)
check("Today:" in s and "Tomorrow:" in s, "multi-day summary groups by day")

# 5. khal failures surface as a sentence, not an exception
real = ct.subprocess.run
def no_khal(*a, **k): raise FileNotFoundError()
ct.subprocess.run = no_khal
check(ct.get_calendar("today") == "Could not read the calendar: khal is not installed", "missing khal handled")
ct.subprocess.run = real

print("\nALL PASSED" if ok else "\nFAILURES")
sys.exit(0 if ok else 1)
