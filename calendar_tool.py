"""
calendar_tool.py — read the local khal calendar for Jarvis.

Events come from `khal list --json ...`, so this works offline against the
vdir store under ~/.local/share/calendars (no Google sign-in involved).
Creating events goes through `khal new`; the agent stages an event with
plan_event() and only calls add_event() after the user says yes.

Public API:
    resolve_when(text, today)  -> (start_date, days, label)
    fetch_events(start, days)  -> [Event, ...]
    find_conflicts(events)     -> [(Event, Event), ...]
    get_calendar(when)         -> speech-friendly summary string (the tool)
    parse_event_time(text, today, now) -> EventPlan  (date + times, no khal)
    plan_event(title, when)    -> EventPlan with clashes filled in
    add_event(plan)            -> spoken confirmation; raises CalendarError
"""
import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from datetime import date, datetime, time as dtime, timedelta

KHAL_TIMEOUT = 10          # seconds
DEFAULT_EVENT_MINUTES = 60
# Calendar new events go to; empty = khal's own default_calendar
NEW_EVENT_CALENDAR = os.getenv("CALENDAR_DEFAULT", "").strip()
# Extra args for every khal call; tests point this at a throwaway config
KHAL_ARGS = os.getenv("KHAL_ARGS", "").split()
MAX_EVENTS_SPOKEN = 12     # a week of events read aloud is unlistenable past this

_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}


@dataclass
class Event:
    day: date
    title: str
    start: str = ""        # "HH:MM", empty for all-day events
    end: str = ""
    calendar: str = ""
    all_day: bool = False

    def minutes(self):
        """(start, end) in minutes since midnight, or None for all-day events."""
        if self.all_day or not self.start:
            return None
        s = _to_minutes(self.start)
        e = _to_minutes(self.end) if self.end else s
        if e <= s:              # ends after midnight, or zero-length
            e = 24 * 60 if e < s else s
        return s, e


def _to_minutes(hhmm):
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def _spoken_time(hhmm):
    """'08:30' -> '8:30 AM', '17:00' -> '5 PM'."""
    h, m = (int(x) for x in hhmm.split(":"))
    suffix = "AM" if h < 12 else "PM"
    h12 = h % 12 or 12
    return f"{h12} {suffix}" if m == 0 else f"{h12}:{m:02d} {suffix}"


def _spoken_day(d, today):
    if d == today:
        return "Today"
    if d == today + timedelta(days=1):
        return "Tomorrow"
    return f"{d.strftime('%A')} {d.day} {d.strftime('%B')}"


# ── "when" parsing ──────────────────────────────────────────────────────────

def resolve_when(text, today=None):
    """Turn a spoken range ('tomorrow', 'friday', 'this week', '20 november',
    '2026-11-20') into (start_date, number_of_days, label).
    Anything unrecognised means today."""
    today = today or date.today()
    t = re.sub(r"[^\w\s-]", " ", (text or "").lower()).strip()
    t = re.sub(r"\s+", " ", t)

    if not t or t in {"today", "now", "when", "day", "date"} or "today" in t or "tonight" in t:
        return today, 1, "Today"
    if "day after tomorrow" in t:
        d = today + timedelta(days=2)
        return d, 1, _spoken_day(d, today)
    if "tomorrow" in t:
        return today + timedelta(days=1), 1, "Tomorrow"
    if "yesterday" in t:
        d = today - timedelta(days=1)
        return d, 1, "Yesterday"

    if "next week" in t:
        start = today + timedelta(days=7 - today.weekday())
        return start, 7, "Next week"
    if "weekend" in t:
        start = today + timedelta(days=(5 - today.weekday()) % 7)
        if today.weekday() == 6:        # already Sunday: just today
            return today, 1, "This weekend"
        return start, 2, "This weekend"
    if "week" in t:
        return today, 7 - today.weekday(), "This week"

    m = re.search(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", t)
    if m:
        try:
            d = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            return d, 1, _spoken_day(d, today)
        except ValueError:
            pass

    # "20 november", "20th of november", "november 20"
    month_re = "|".join(sorted(_MONTHS, key=len, reverse=True))
    m = (re.search(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?(?: of)? ({month_re})\b", t)
         or re.search(rf"\b({month_re}) (\d{{1,2}})(?:st|nd|rd|th)?\b", t))
    if m:
        a, b = m.group(1), m.group(2)
        day_num, month = (int(a), _MONTHS[b]) if a.isdigit() else (int(b), _MONTHS[a])
        try:
            d = date(today.year, month, day_num)
            if d < today - timedelta(days=31):   # "5 january" said in November = next year
                d = date(today.year + 1, month, day_num)
            return d, 1, _spoken_day(d, today)
        except ValueError:
            pass

    for i, name in enumerate(_WEEKDAYS):
        if re.search(rf"\b{name}\b", t) or re.search(rf"\b{name[:3]}\b", t):
            ahead = (i - today.weekday()) % 7
            if ahead == 0 and "next" in t:
                ahead = 7
            d = today + timedelta(days=ahead)
            return d, 1, _spoken_day(d, today)

    return today, 1, "Today"


# ── khal access ─────────────────────────────────────────────────────────────

class CalendarError(RuntimeError):
    pass


def fetch_events(start, days=1):
    """Events for `days` days starting at `start`, in chronological order.

    khal's --json output carries no day marker and skips empty days, so a
    multi-day range is asked for one day at a time (each call is ~0.1s)."""
    events = []
    for offset in range(days):
        day = start + timedelta(days=offset)
        events.extend(_to_event(row, day) for row in _khal_day(day))
    return events


def _khal_day(day):
    cmd = ["khal", *KHAL_ARGS, "list"]
    for field in ("title", "start-time", "end-time", "calendar", "all-day"):
        cmd += ["--json", field]
    cmd += [day.isoformat(), "1d"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=KHAL_TIMEOUT)
    except FileNotFoundError:
        raise CalendarError("khal is not installed")
    except subprocess.TimeoutExpired:
        raise CalendarError("khal timed out")
    if proc.returncode != 0:
        raise CalendarError((proc.stderr or proc.stdout).strip()[:200] or "khal failed")

    rows = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line.startswith("["):
            continue          # "No events" or a warning line
        try:
            rows.extend(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _to_event(row, day):
    return Event(
        day=day,
        title=(row.get("title") or "Untitled").strip(),
        start=(row.get("start-time") or "").strip(),
        end=(row.get("end-time") or "").strip(),
        calendar=(row.get("calendar") or "").strip(),
        all_day=str(row.get("all-day", "")).lower() == "true",
    )


# ── conflicts ───────────────────────────────────────────────────────────────

def find_conflicts(events):
    """Pairs of timed events on the same day whose times overlap."""
    clashes = []
    timed = [e for e in events if e.minutes()]
    for i, a in enumerate(timed):
        for b in timed[i + 1:]:
            if a.day != b.day:
                continue
            (s1, e1), (s2, e2) = a.minutes(), b.minutes()
            if s1 < e2 and s2 < e1:
                clashes.append((a, b))
    return clashes


# ── the tool ────────────────────────────────────────────────────────────────

def _describe(e):
    if e.all_day or not e.start:
        return f"all day: {e.title}"
    if e.end and e.end != e.start:
        return f"{_spoken_time(e.start)} to {_spoken_time(e.end)}: {e.title}"
    return f"{_spoken_time(e.start)}: {e.title}"


def summarize(events, label, today=None, multi_day=False):
    """Plain-text summary the LLM can read back. Kept separate from khal access
    so it can be unit-tested."""
    today = today or date.today()
    if not events:
        return f"{label}: nothing on the calendar."

    shown = events[:MAX_EVENTS_SPOKEN]
    parts = [f"{label}: {len(events)} event{'s' if len(events) != 1 else ''}."]
    if multi_day:
        current = None
        for e in shown:
            if e.day != current:
                current = e.day
                parts.append(f"{_spoken_day(e.day, today)}:")
            parts.append(_describe(e) + ".")
    else:
        parts.extend(_describe(e) + "." for e in shown)
    if len(events) > len(shown):
        parts.append(f"And {len(events) - len(shown)} more.")

    clashes = find_conflicts(events)
    if clashes:
        bits = []
        for a, b in clashes[:3]:
            where = f" on {_spoken_day(a.day, today)}" if multi_day else ""
            bits.append(f"'{a.title}' overlaps '{b.title}'{where}")
        parts.append("CLASHES: " + "; ".join(bits) + ".")
    return " ".join(parts)


def get_calendar(when=""):
    """Tool: what is on the calendar for today / tomorrow / a weekday / a date / this week."""
    start, days, label = resolve_when(when)
    try:
        events = fetch_events(start, days)
    except CalendarError as e:
        return f"Could not read the calendar: {e}"
    return summarize(events, label, multi_day=days > 1)


# ── creating events ─────────────────────────────────────────────────────────

class EventParseError(ValueError):
    pass


@dataclass
class EventPlan:
    title: str
    day: date
    start: str = ""            # "HH:MM"; empty = all-day
    end: str = ""
    calendar: str = ""
    notes: list = field(default_factory=list)      # assumptions to read back
    clashes: list = field(default_factory=list)    # titles of overlapping events
    specified: bool = False    # the text named a day or a time (not just "today" by default)

    @property
    def all_day(self):
        return not self.start

    def spoken(self, today=None):
        today = today or date.today()
        when = _spoken_day(self.day, today)
        if self.all_day:
            text = f"'{self.title}', {when}, all day"
        else:
            text = f"'{self.title}', {when} from {_spoken_time(self.start)} to {_spoken_time(self.end)}"
        if self.calendar:
            text += f", on your {self.calendar} calendar"
        return text


_MERIDIEM = r"(a\.?m\.?|p\.?m\.?)"
_RANGE_RE = re.compile(
    rf"\b(?:from\s+)?(\d{{1,2}})(?:[:.](\d{{2}}))?\s*{_MERIDIEM}?\s*(?:to|till|until|-|–)\s*"
    rf"(\d{{1,2}})(?:[:.](\d{{2}}))?\s*{_MERIDIEM}?(?!\s*(?:hours?|hrs?|minutes?|mins?))", re.I)
_CLOCK_RE = re.compile(rf"\b(?:at\s+)?(\d{{1,2}})[:.](\d{{2}})\s*{_MERIDIEM}?", re.I)
_AMPM_RE = re.compile(rf"\b(?:at\s+)?(\d{{1,2}})\s*{_MERIDIEM}", re.I)
_AT_RE = re.compile(r"\bat\s+(\d{1,2})\b(?!\s*(?:st|nd|rd|th)\b)(?![:.]\d)", re.I)
_WORD_TIME_RE = re.compile(r"\b(?:at\s+)?(noon|midday|midnight)\b", re.I)
_DURATION_RE = re.compile(
    r"\bfor\s+(an?|one|two|three|four|half an?|\d+(?:\.\d+)?)\s*(hours?|hrs?|minutes?|mins?)\b", re.I)
_DURATION_WORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4}
# Spans that name a date or time; removed when the title has to be recovered
# from one combined phrase ("team sync tomorrow at 4 pm").
_WHEN_WORDS_RE = re.compile(
    r"\b(?:(?:on|this|next)\s+)?(?:today|tonight|tomorrow|day after tomorrow|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"mon|tue|tues|wed|thu|thurs|fri|sat|sun)\b|"
    r"\b(?:on\s+)?\d{4}-\d{1,2}-\d{1,2}\b|"
    r"\b(?:on\s+)?(?:the\s+)?\d{1,2}(?:st|nd|rd|th)?(?:\s+of)?\s+(?:" + "|".join(sorted(_MONTHS, key=len, reverse=True)) + r")\b|"
    r"\b(?:on\s+)?(?:" + "|".join(sorted(_MONTHS, key=len, reverse=True)) + r")\s+\d{1,2}(?:st|nd|rd|th)?\b",
    re.I)


def _hm(hour, minute, meridiem, notes, guess=True):
    """24h (hour, minute). Without am/pm, 1-7 is read as PM and 8-11 as AM
    (when people schedule things), and that guess is noted for read-back."""
    h, m = int(hour), int(minute or 0)
    if not (0 <= h <= 23 and 0 <= m <= 59):
        raise EventParseError(f"{hour}:{minute or '00'} is not a valid time")
    if meridiem:
        pm = meridiem.lower().startswith("p")
        if h > 12 or h == 0:
            raise EventParseError(f"{hour} {meridiem} is not a valid time")
        h = (h % 12) + (12 if pm else 0)
    elif guess and 1 <= h <= 7:
        h += 12
        notes.append(f"I took {hour} to mean {_spoken_time(f'{h:02d}:{m:02d}')}")
    return h, m


def _fmt(h, m):
    return f"{h:02d}:{m:02d}"


def parse_event_time(text, today=None, now=None):
    """'tomorrow at 4 pm', 'friday 3 to 5 pm', '20 november 10:30 for 2 hours',
    'next monday' (all day) -> EventPlan without title or calendar.
    Raises EventParseError with a speakable reason."""
    now = now or datetime.now()
    today = today or now.date()
    raw = text or ""
    t = " ".join(raw.lower().split())
    notes = []

    day, days, label = resolve_when(t, today)
    if days > 1:
        raise EventParseError(f"{label.lower()} is more than one day; say which day")
    explicit_day = not (label == "Today" and "today" not in t and "tonight" not in t)

    start = end = None
    m = _RANGE_RE.search(t)
    if m:
        h1, m1, ap1, h2, m2, ap2 = m.groups()
        # "3 to 5 pm": the start borrows the end's am/pm when that keeps it earlier
        if ap2 and not ap1:
            cand = _hm(h1, m1, ap2, [])
            end_hm = _hm(h2, m2, ap2, notes)
            if cand < end_hm:
                start = cand
            else:
                start = _hm(h1, m1, None, notes, guess=False)
                if start[0] < 8:
                    notes.append(f"I took {h1} to mean {_spoken_time(_fmt(*start))}")
            end = end_hm
        else:
            start = _hm(h1, m1, ap1, notes)
            end = _hm(h2, m2, ap2 or ap1, notes, guess=not ap1)
            if end <= start and not (ap2 or ap1) and end[0] < 12:
                end = (end[0] + 12, end[1])
    else:
        for rx, has_min in ((_CLOCK_RE, True), (_AMPM_RE, False)):
            m = rx.search(t)
            if m:
                g = m.groups()
                start = _hm(g[0], g[1] if has_min else 0, g[2] if has_min else g[1], notes,
                            guess=not (has_min and int(g[0]) >= 13))
                break
        if start is None:
            m = _WORD_TIME_RE.search(t)
            if m:
                start = (0, 0) if m.group(1).lower() == "midnight" else (12, 0)
        if start is None:
            m = _AT_RE.search(t)
            if m:
                start = _hm(m.group(1), 0, None, notes)

    if start is None:
        return EventPlan(title="", day=day, notes=notes, specified=explicit_day)   # all-day

    if end is None:
        minutes = DEFAULT_EVENT_MINUTES
        d = _DURATION_RE.search(t)
        if d:
            qty, unit = d.group(1).lower(), d.group(2).lower()
            n = 0.5 if qty.startswith("half") else _DURATION_WORDS.get(qty) or float(qty)
            minutes = int(n * (60 if unit.startswith("h") else 1))
        total = start[0] * 60 + start[1] + max(minutes, 5)
        if total >= 24 * 60:
            total = 23 * 60 + 59
            notes.append("it would run past midnight, so I end it at 11:59 PM")
        end = divmod(total, 60)
    if end <= start:
        raise EventParseError("the end time is before the start time")

    if day == now.date() and start < (now.hour, now.minute):
        if explicit_day:
            notes.append("that time has already passed today")
        else:
            day = day + timedelta(days=1)
            notes.append("that time has passed today, so I put it tomorrow")

    return EventPlan(title="", day=day, start=_fmt(*start), end=_fmt(*end), notes=notes, specified=True)


def strip_when(text):
    """Remove date/time phrases, leaving a likely event title."""
    t = text or ""
    for rx in (_RANGE_RE, _DURATION_RE, _CLOCK_RE, _AMPM_RE, _WORD_TIME_RE, _AT_RE, _WHEN_WORDS_RE):
        t = rx.sub(" ", t)
    t = re.sub(r"^\s*(?:schedule|add|create|put|book|set up|make)\s+(?:an?\s+)?(?:event\s+|meeting\s+)?(?:for\s+|called\s+)?",
               "", t, flags=re.I)
    t = re.sub(r"\b(?:on|at|from|for)\s*$", "", t.strip(), flags=re.I)
    return " ".join(t.split()).strip(" ,.-")


def default_calendar():
    return NEW_EVENT_CALENDAR


def plan_event(title, when, today=None, now=None, user_text=""):
    """Parse and check an event without creating it. Raises EventParseError.

    user_text is the user's own utterance. Its date and time win over `when`,
    which the LLM may have paraphrased ("friday 3 to 5 pm" -> "friday at 3 pm")."""
    title = " ".join((title or "").split())
    plan = None
    if user_text:
        try:
            plan = parse_event_time(user_text, today=today, now=now)
        except EventParseError:
            plan = None
        if plan is not None and not plan.specified:
            plan = None
    if plan is None:
        plan = parse_event_time(when or title, today=today, now=now)
    if not title or title.lower() == (when or "").strip().lower():
        title = strip_when(title or when)
    if not title or title.lower() in {"title", "event", "when", "name"}:
        raise EventParseError("I need a name for the event")
    plan.title = title[:120]
    plan.calendar = default_calendar()
    try:
        same_day = fetch_events(plan.day, 1)
    except CalendarError:
        same_day = []
    probe = Event(day=plan.day, title=plan.title, start=plan.start, end=plan.end, all_day=plan.all_day)
    plan.clashes = [b.title if a is probe else a.title
                    for a, b in find_conflicts(same_day + [probe]) if probe in (a, b)]
    return plan


def add_event(plan):
    """Write the event with `khal new`. Returns a spoken confirmation."""
    cmd = ["khal", *KHAL_ARGS, "new"]
    if plan.calendar:
        cmd += ["-a", plan.calendar]
    cmd.append(plan.day.isoformat())
    if not plan.all_day:
        cmd += [plan.start, plan.end]
    # "--" stops a title starting with "-" being read as an option
    cmd += ["--", plan.title]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=KHAL_TIMEOUT)
    except FileNotFoundError:
        raise CalendarError("khal is not installed")
    except subprocess.TimeoutExpired:
        raise CalendarError("khal timed out")
    if proc.returncode != 0:
        raise CalendarError((proc.stderr or proc.stdout).strip()[:200] or "khal new failed")
    return f"Added {plan.spoken()}."


if __name__ == "__main__":
    import sys
    print(get_calendar(" ".join(sys.argv[1:])))
