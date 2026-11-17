"""
calendar_tool.py — read the local khal calendar for Jarvis.

Events come from `khal list --json ...`, so this works offline against the
vdir store under ~/.local/share/calendars (no Google sign-in involved).
Read-only: nothing here creates or edits events.

Public API:
    resolve_when(text, today)  -> (start_date, days, label)
    fetch_events(start, days)  -> [Event, ...]
    find_conflicts(events)     -> [(Event, Event), ...]
    get_calendar(when)         -> speech-friendly summary string (the tool)
"""
import json
import re
import subprocess
from dataclasses import dataclass
from datetime import date, datetime, timedelta

KHAL_TIMEOUT = 10          # seconds
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
    cmd = ["khal", "list"]
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


if __name__ == "__main__":
    import sys
    print(get_calendar(" ".join(sys.argv[1:])))
