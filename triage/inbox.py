"""受信箱（.emlファイル）の読み込み。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from email import policy
from email.parser import BytesParser
from email.utils import parseaddr, parsedate_to_datetime
from pathlib import Path


@dataclass(frozen=True)
class Mail:
    message_id: str
    received_at: datetime
    from_name: str
    from_addr: str
    subject: str
    body: str


def read_mail(path: Path) -> Mail:
    message = BytesParser(policy=policy.default).parsebytes(path.read_bytes())
    name, addr = parseaddr(str(message["From"]))
    body_part = message.get_body(preferencelist=("plain",))
    return Mail(
        # Message-IDは二重取り込みを防ぐ鍵にする。無い場合はファイル名で代用する。
        message_id=str(message["Message-ID"] or f"<{path.name}>").strip(),
        received_at=parsedate_to_datetime(str(message["Date"])),
        from_name=name,
        from_addr=addr,
        subject=str(message["Subject"] or "（件名なし）"),
        body=body_part.get_content().strip() if body_part else "",
    )


def read_inbox(inbox_dir: Path) -> list[Mail]:
    return [read_mail(path) for path in sorted(inbox_dir.glob("*.eml"))]
