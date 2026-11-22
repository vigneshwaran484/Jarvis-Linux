"""
test_mail.py — header decoding, the unread summary, and the IMAP flow of
mail_tool against a fake server. No network or credentials needed.

Run:  .venv/bin/python test_mail.py
"""
import imaplib
import sys

import mail_tool as mt
from mail_tool import Mail

ok = True
def check(cond, msg):
    global ok
    print(("✅ " if cond else "❌ ") + msg)
    ok &= bool(cond)


# 1. header parsing
m = mt.parse_headers(b"From: Priya Raman <priya@example.com>\r\nSubject: Interview slot\r\n\r\n")
check((m.sender, m.subject) == ("Priya Raman", "Interview slot"), "plain headers")
m = mt.parse_headers(b"From: noreply@github.com\r\nSubject: =?UTF-8?B?UFIgbWVyZ2VkIOKckw==?=\r\n\r\n")
check(m.sender == "noreply" and m.subject == "PR merged ✓", "bare address + RFC 2047 subject")
m = mt.parse_headers(b"From: =?utf-8?q?Caf=C3=A9_Team?= <x@y.z>\r\n\r\n")
check((m.sender, m.subject) == ("Café Team", "no subject"), "encoded sender, missing subject")
m = mt.parse_headers(b"From: A <a@b.c>\r\nSubject: line one\r\n  line two\r\n\r\n")
check(m.subject == "line one line two", "folded subject is unfolded")

# 2. summary
check(mt.summarize(0, []) == "Inbox: no unread email.", "empty inbox")
check(mt.summarize(1, [Mail("Priya", "Hi")]) == "Inbox: 1 unread email. Priya: Hi.", "one mail")
s = mt.summarize(12, [Mail("A", "x"), Mail("B", "y")])
check(s == "Inbox: 12 unread emails. Newest: A: x; B: y.", "more unread than read aloud")


# 3. IMAP flow against a fake
class FakeIMAP:
    def __init__(self, ids=b"1 2 3 4", fail_login=False):
        self.ids, self.fail_login = ids, fail_login
        self.calls = []
    def login(self, user, pw):
        self.calls.append(("login", user, pw))
        if self.fail_login:
            raise imaplib.IMAP4.error("AUTHENTICATIONFAILED")
    def select(self, box, readonly=False):
        self.calls.append(("select", box, readonly))
        return "OK", [b"4"]
    def search(self, charset, crit):
        self.calls.append(("search", crit))
        return "OK", [self.ids]
    def fetch(self, msg_id, spec):
        self.calls.append(("fetch", msg_id, spec))
        n = msg_id.decode()
        return "OK", [(b"hdr", f"From: Sender{n} <s{n}@x.y>\r\nSubject: Mail {n}\r\n\r\n".encode()), b")"]
    def logout(self):
        self.calls.append(("logout",))

mt.MAIL_ADDRESS, mt.MAIL_APP_PASSWORD = "me@example.com", "abcd efgh ijkl mnop"
fake = FakeIMAP()
total, mails = mt.fetch_unread(limit=2, _imap_factory=lambda: fake)
check(total == 4, "counts every unread message")
check([x.subject for x in mails] == ["Mail 4", "Mail 3"], "returns only the newest, newest first")
check(("login", "me@example.com", "abcdefghijklmnop") in fake.calls, "spaces stripped from the app password")
check(("select", "INBOX", True) in fake.calls, "inbox opened read-only")
check(all("PEEK" in c[2] for c in fake.calls if c[0] == "fetch"), "fetch uses PEEK (does not mark as read)")
check(fake.calls[-1] == ("logout",), "always logs out")

fake = FakeIMAP(ids=b"")
check(mt.fetch_unread(_imap_factory=lambda: fake) == (0, []), "no unread mail")

fake = FakeIMAP(fail_login=True)
try:
    mt.fetch_unread(_imap_factory=lambda: fake); raised = False
except mt.MailError as e:
    raised = "login rejected" in str(e)
check(raised and fake.calls[-1] == ("logout",), "bad password -> MailError, still logs out")

def unreachable(): raise OSError("Network is unreachable")
try:
    mt.fetch_unread(_imap_factory=unreachable); raised = False
except mt.MailError:
    raised = True
check(raised, "offline -> MailError")

# 4. tool-level messages
mt.MAIL_ADDRESS = ""
check("not set up" in mt.get_unread_email(), "unconfigured tool explains how to enable it")

print("\nALL PASSED" if ok else "\nFAILURES")
sys.exit(0 if ok else 1)
