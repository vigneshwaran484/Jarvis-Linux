"""
mail_tool.py — read-only unread-mail summary for Jarvis over IMAP.

Works with Gmail (or any IMAP mailbox) using an app password, so there is no
OAuth client to register. Messages are fetched with BODY.PEEK, which leaves
them unread, and only the From/Subject headers are downloaded: bodies never
leave the mail server. With a local LLM the headers never leave this machine.

Setup (Gmail): turn on 2-step verification, create an app password at
https://myaccount.google.com/apppasswords, then put in .env:
    MAIL_ADDRESS=you@gmail.com
    MAIL_APP_PASSWORD=abcd efgh ijkl mnop

Public API:
    configured()            -> bool
    fetch_unread(limit)     -> (total_unread, [Mail, ...])  newest first
    get_unread_email()      -> speech-friendly summary string (the tool)
"""
import email
import imaplib
import re
import socket
from dataclasses import dataclass
from email.header import decode_header, make_header
from email.utils import parseaddr

from config import MAIL_ADDRESS, MAIL_APP_PASSWORD, MAIL_IMAP_HOST, MAIL_MAX_SPOKEN

IMAP_TIMEOUT = 10   # seconds


@dataclass
class Mail:
    sender: str
    subject: str


class MailError(RuntimeError):
    pass


def configured():
    return bool(MAIL_ADDRESS and MAIL_APP_PASSWORD)


def _decode(value):
    """RFC 2047 header ('=?UTF-8?B?...?=') -> plain text."""
    if not value:
        return ""
    try:
        text = str(make_header(decode_header(value)))
    except Exception:
        text = str(value)
    return re.sub(r"\s+", " ", text).strip()


def parse_headers(raw_bytes):
    """Raw 'From:/Subject:' header block -> Mail."""
    msg = email.message_from_bytes(raw_bytes)
    name, addr = parseaddr(_decode(msg.get("From", "")))
    sender = name or addr.split("@")[0] or "unknown sender"
    subject = _decode(msg.get("Subject", "")) or "no subject"
    return Mail(sender=sender, subject=subject)


def fetch_unread(limit=None, _imap_factory=None):
    """(total unread in the inbox, the newest `limit` of them as Mail objects)."""
    if not configured():
        raise MailError("not configured")
    limit = limit or MAIL_MAX_SPOKEN
    factory = _imap_factory or (lambda: imaplib.IMAP4_SSL(MAIL_IMAP_HOST, timeout=IMAP_TIMEOUT))
    try:
        conn = factory()
    except (OSError, socket.timeout) as e:
        raise MailError(f"cannot reach {MAIL_IMAP_HOST}: {e}")
    try:
        try:
            conn.login(MAIL_ADDRESS, MAIL_APP_PASSWORD.replace(" ", ""))
        except imaplib.IMAP4.error:
            raise MailError("login rejected; check MAIL_ADDRESS and the app password")
        conn.select("INBOX", readonly=True)          # readonly: nothing gets marked as read
        status, data = conn.search(None, "UNSEEN")
        if status != "OK":
            raise MailError("search failed")
        ids = data[0].split() if data and data[0] else []
        mails = []
        for msg_id in reversed(ids[-limit:]):        # newest first
            status, parts = conn.fetch(msg_id, "(BODY.PEEK[HEADER.FIELDS (FROM SUBJECT)])")
            if status != "OK":
                continue
            for part in parts:
                if isinstance(part, tuple) and len(part) > 1:
                    mails.append(parse_headers(part[1]))
                    break
        return len(ids), mails
    except (OSError, socket.timeout, imaplib.IMAP4.abort) as e:
        raise MailError(f"connection problem: {e}")
    finally:
        try:
            conn.logout()
        except Exception:
            pass


def summarize(total, mails):
    if total == 0:
        return "Inbox: no unread email."
    parts = [f"Inbox: {total} unread email{'s' if total != 1 else ''}."]
    lead = "Newest: " if total > len(mails) else ""
    items = [f"{m.sender}: {m.subject}" for m in mails]
    if items:
        parts.append(lead + "; ".join(items) + ".")
    return " ".join(parts)


def get_unread_email(dummy=None):
    """Tool: how many unread emails there are and who the newest ones are from."""
    if not configured():
        return ("Email is not set up. Add MAIL_ADDRESS and MAIL_APP_PASSWORD to the .env file "
                "to enable it.")
    try:
        total, mails = fetch_unread()
    except MailError as e:
        return f"Could not check email: {e}"
    return summarize(total, mails)


if __name__ == "__main__":
    print(get_unread_email())
