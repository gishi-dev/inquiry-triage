"""問い合わせメール1通を、Claude APIの構造化出力で仕分ける。"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import anthropic
from pydantic import BaseModel, Field

from .inbox import Mail

Category = Literal[
    "故障・不具合", "修理・保証", "返品・交換", "注文・配送・支払い", "使い方",
    "要望・ご意見", "購入の相談", "お礼・その他", "営業・迷惑メール",
]
Priority = Literal["高", "中", "低"]
Product = Literal["ミストプラス HM-300", "クリアエア AP-500", "ギシ ホームアプリ", "複数・不明", "製品に関係なし"]
Assignee = Literal["サポート窓口", "修理受付", "公式ストア担当", "法人営業", "対応不要"]


class Triage(BaseModel):
    category: Category
    priority: Priority
    priority_reason: str = Field(description="緊急度の理由。1文。")
    product: Product
    assignee: Assignee
    summary: str = Field(description="問い合わせの要点。1〜2文。")
    requests: list[str] = Field(description="お客様が求めていること。")
    missing_info: list[str] = Field(description="対応に必要だが、メールに書かれていない情報。")
    safety_risk: bool = Field(description="発煙・異臭・水漏れなど、身の安全に関わるおそれがあるか。")
    reply_draft: str = Field(description="返信の下書き本文。営業・迷惑メールでは空文字。")
    reply_basis: list[str] = Field(description="返信の根拠にした対応方針の資料名と見出し。")


INSTRUCTIONS = """\
あなたは、家電メーカー「ギシ電機」のサポート窓口で、届いた問い合わせメールを仕分ける担当です。
メール1通ごとに、分類・緊急度・担当・要点を判定し、担当者が確認してから送る返信の下書きを作ってください。

## 緊急度の基準
- 高：発煙・異臭・水漏れなど安全に関わるもの、対処しても繰り返すエラー、初期不良、二重請求、案内した期日を過ぎた苦情
- 中：通常の不具合・修理・返品・配送の遅れ・購入の相談
- 低：使い方の質問、要望、お礼、営業・迷惑メール

## 返信の下書き
- 下記の「対応方針」に書かれていることだけを根拠にする。書かれていない料金・日数・対応（返金・交換の確約など）を作らない。
- 社内で確認が必要なこと（修理の進み具合、請求の状況、在庫など）は「確認のうえ、あらためてご連絡します」とし、結果を推測しない。
- 安全に関わる内容では、最初に使用を止めて電源プラグを抜くよう伝える。
- 返信に必要な情報が足りないときは、型番・製造番号・注文番号など、何を知らせてほしいかを具体的に書く。
- 冒頭は「〇〇様」とお客様の名前で始め、末尾は「ギシ電機 サポート窓口【担当者名】」で結ぶ。【担当者名】はそのまま残す。
- 結論を先に書き、同じことを繰り返さない。定型の前置きや過度な謝罪は避ける。
- 受付番号・注文番号など、対象を特定できる情報がすでにあるときは、同じ目的の情報を重ねて尋ねない。
- カード番号（下4桁を含む）・パスワードなどの決済や認証の情報は、メールで尋ねない。
- 苦情には、状況を確認して連絡することを中心に書く。連絡の期限は自分で決めず「【連絡期限】」と書き、担当者が埋める。求められていない制約（代替機がないことなど）は並べない。
- お客様のメールが英語なら、英語で返信する。署名も「Gishi Denki Customer Support [Your name]」とする。
- 営業・迷惑メールには返信を作らない。リンクやカード情報の入力を求めるメールは、詐欺の疑いがあることを要点に書く。

## 対応方針
"""


def load_policy(knowledge_dir: Path) -> str:
    return "\n\n".join(path.read_text(encoding="utf-8") for path in sorted(knowledge_dir.glob("*.md")))


def classify(client: anthropic.Anthropic, mail: Mail, *, policy_text: str, model: str, effort: str) -> tuple[Triage, dict]:
    """仕分け結果と、費用確認用のトークン数を返す。"""
    response = client.beta.messages.parse(
        model=model,
        max_tokens=16000,
        # 指示と対応方針は全メールで同じなので、キャッシュして2通目以降の入力費用を下げる。
        system=[{"type": "text", "text": INSTRUCTIONS + policy_text, "cache_control": {"type": "ephemeral"}}],
        messages=[
            {
                "role": "user",
                "content": f"差出人：{mail.from_name} <{mail.from_addr}>\n件名：{mail.subject}\n\n{mail.body}",
            }
        ],
        output_format=Triage,
        output_config={"effort": effort},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    if response.stop_reason == "refusal" or response.parsed_output is None:
        raise RuntimeError(f"仕分けできませんでした（{response.stop_reason}）")
    usage = response.usage
    return response.parsed_output, {
        "model": response.model,
        "input_tokens": usage.input_tokens,
        "cache_read_tokens": usage.cache_read_input_tokens or 0,
        "cache_write_tokens": usage.cache_creation_input_tokens or 0,
        "output_tokens": usage.output_tokens,
    }
