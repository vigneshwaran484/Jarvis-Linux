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

# 6. event creation: date/time parsing (no khal)
from datetime import datetime
NOW = datetime(2026, 11, 17, 13, 40)        # a Tuesday afternoon
TUE, WED, FRI20 = date(2026, 11, 17), date(2026, 11, 18), date(2026, 11, 20)
ev_cases = [
    ("tomorrow at 4 pm",              (WED, "16:00", "17:00")),
    ("tomorrow at 4",                 (WED, "16:00", "17:00")),   # 1-7 without am/pm = PM
    ("friday 3 to 5 pm",              (FRI20, "15:00", "17:00")),
    ("friday 10 to 2 pm",             (FRI20, "10:00", "14:00")),
    ("20 november 10:30 for 2 hours", (FRI20, "10:30", "12:30")),
    ("tomorrow 4:30pm for 30 minutes",(WED, "16:30", "17:00")),
    ("wednesday at noon",             (WED, "12:00", "13:00")),
    ("tomorrow 9.30 am",              (WED, "09:30", "10:30")),
    ("at 9 am",                       (WED, "09:00", "10:00")),   # passed today, no day said -> tomorrow
    ("today at 9am",                  (TUE, "09:00", "10:00")),   # said today: kept, with a note
    ("next monday",                   (date(2026, 11, 23), "", "")),
]
for text, want in ev_cases:
    p = ct.parse_event_time(text, now=NOW)
    check((p.day, p.start, p.end) == want, f"event time {text!r} -> {p.day} {p.start}-{p.end}")
check(ct.parse_event_time("today at 9am", now=NOW).notes == ["that time has already passed today"],
      "past time today is read back")
for bad in ("this week at 4", "tomorrow at 25"):
    try:
        ct.parse_event_time(bad, now=NOW); check(False, f"{bad!r} should be rejected")
    except ct.EventParseError:
        check(True, f"{bad!r} rejected")
for text, title in [("schedule a team sync tomorrow at 4 PM", "team sync"),
                    ("Lab review on 20 november at 10:30 for 2 hours", "Lab review"),
                    ("gym at 6", "gym")]:
    check(ct.strip_when(text) == title, f"title from {text!r} -> {ct.strip_when(text)!r}")
# the user's own words beat a paraphrased `when` from the model
ct.fetch_events = lambda start, days=1: []
p = ct.plan_event("dentist", "friday at 3 pm", now=NOW, user_text="add dentist on friday 3 to 5 pm")
check((p.start, p.end) == ("15:00", "17:00"), "user's '3 to 5 pm' wins over model's 'at 3 pm'")

print("\nALL PASSED" if ok else "\nFAILURES")
sys.exit(0 if ok else 1)
