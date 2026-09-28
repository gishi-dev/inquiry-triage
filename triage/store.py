"""メールと仕分け結果の保存（SQLite）。"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path

from .classify import Triage
from .inbox import Mail

STATUSES = {"new": "未処理", "open": "要対応", "done": "対応済み", "skipped": "対応不要"}

_SCHEMA = """create table if not exists mails (
    id integer primary key,
    message_id text not null unique,
    received_at text not null,
    from_name text not null,
    from_addr text not null,
    subject text not null,
    body text not null,
    status text not null default 'new',
    triage text,
    reply text not null default '',
    usage text,
    processed_at text,
    done_at text
)"""


class Store:
    def __init__(self, path: Path) -> None:
        self.path = path

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row
        db.execute(_SCHEMA)
        return db

    def import_mails(self, mails: list[Mail]) -> int:
        """新しいメールだけを登録する。Message-IDが同じメールは、何度取り込んでも1件のまま。"""
        with closing(self._connect()) as db, db:
            before = db.total_changes
            db.executemany(
                "insert or ignore into mails (message_id, received_at, from_name, from_addr, subject, body) "
                "values (?, ?, ?, ?, ?, ?)",
                [
                    (m.message_id, m.received_at.isoformat(), m.from_name, m.from_addr, m.subject, m.body)
                    for m in mails
                ],
            )
            return db.total_changes - before

    def unprocessed(self) -> list[tuple[int, Mail]]:
        with closing(self._connect()) as db:
            rows = db.execute("select * from mails where triage is null order by received_at").fetchall()
        return [(row["id"], _mail(row)) for row in rows]

    def save_triage(self, mail_id: int, triage: Triage, usage: dict) -> bool:
        """まだ仕分けていない行にだけ書く。同時に処理しても、先に書いた結果を上書きしない。"""
        status = "skipped" if triage.assignee == "対応不要" else "open"
        with closing(self._connect()) as db, db:
            cursor = db.execute(
                "update mails set triage = ?, reply = ?, usage = ?, status = ?, processed_at = ? "
                "where id = ? and triage is null",
                (
                    triage.model_dump_json(),
                    triage.reply_draft,
                    json.dumps(usage),
                    status,
                    datetime.now().isoformat(timespec="seconds"),
                    mail_id,
                ),
            )
            return cursor.rowcount == 1

    def update_reply(self, mail_id: int, reply: str, done: bool) -> None:
        with closing(self._connect()) as db, db:
            db.execute(
                "update mails set reply = ?, status = ?, done_at = ? where id = ?",
                (
                    reply,
                    "done" if done else "open",
                    datetime.now().isoformat(timespec="seconds") if done else None,
                    mail_id,
                ),
            )

    def all(self) -> list[dict]:
        with closing(self._connect()) as db:
            rows = db.execute("select * from mails order by received_at desc").fetchall()
        return [
            {**dict(row), "triage": Triage.model_validate_json(row["triage"]) if row["triage"] else None}
            for row in rows
        ]

    def usage_total(self) -> dict[str, int]:
        total: dict[str, int] = {}
        with closing(self._connect()) as db:
            for (usage,) in db.execute("select usage from mails where usage is not null"):
                for key, value in json.loads(usage).items():
                    if isinstance(value, int):
                        total[key] = total.get(key, 0) + value
        return total


def _mail(row: sqlite3.Row) -> Mail:
    return Mail(
        row["message_id"],
        datetime.fromisoformat(row["received_at"]),
        row["from_name"],
        row["from_addr"],
        row["subject"],
        row["body"],
    )
