"""問い合わせメール仕分け（デモ）の画面。

起動: streamlit run app.py
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import anthropic
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from triage.classify import classify, load_policy
from triage.inbox import read_inbox
from triage.store import STATUSES, Store

load_dotenv(Path(__file__).parent / ".env")
ROOT = Path(__file__).parent
INBOX_DIR = ROOT / os.getenv("TRIAGE_INBOX_DIR", "inbox")
KNOWLEDGE_DIR = ROOT / "knowledge"
MODEL = os.getenv("TRIAGE_MODEL", "claude-opus-5")
EFFORT = os.getenv("TRIAGE_EFFORT", "medium")
WORKERS = 4
PRIORITY_LABEL = {"高": "🔴 高", "中": "🟡 中", "低": "⚪ 低"}
PRIORITY_ORDER = {"高": 0, "中": 1, "低": 2}


@st.cache_resource
def get_store() -> Store:
    return Store(ROOT / "data" / "mails.sqlite3")


def run_triage(store: Store) -> None:
    targets = store.unprocessed()
    if not targets:
        st.toast("未処理のメールはありません。")
        return
    client = anthropic.Anthropic()
    policy_text = load_policy(KNOWLEDGE_DIR)
    progress = st.progress(0.0, text="仕分けています…")
    failed: list[str] = []

    def job(mail):
        return classify(client, mail, policy_text=policy_text, model=MODEL, effort=EFFORT)

    # 1通目で対応方針をキャッシュに載せてから並列にする。同時に送るとキャッシュが効かない。
    first_id, first_mail = targets[0]
    try:
        store.save_triage(first_id, *job(first_mail))
    except (anthropic.APIError, RuntimeError):
        failed.append(first_mail.subject)
    progress.progress(1 / len(targets), text=f"仕分けています… 1/{len(targets)}")

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(job, mail): (mail_id, mail) for mail_id, mail in targets[1:]}
        for done, future in enumerate(as_completed(futures), 2):
            mail_id, mail = futures[future]
            try:
                store.save_triage(mail_id, *future.result())
            except (anthropic.APIError, RuntimeError):
                failed.append(mail.subject)
            progress.progress(done / len(targets), text=f"仕分けています… {done}/{len(targets)}")
    progress.empty()
    if failed:
        st.error("仕分けできなかったメールがあります（もう一度実行すると再試行します）：" + "、".join(failed))


def sidebar(store: Store, rows: list[dict]) -> None:
    with st.sidebar:
        st.title("問い合わせ仕分け")
        st.caption("架空の家電メーカー「ギシ電機」のサポート窓口に届いたメールで動くデモです。")
        if st.button("受信箱を取り込む", width="stretch"):
            added = store.import_mails(read_inbox(INBOX_DIR))
            st.toast(f"{added}通を取り込みました。" if added else "新しいメールはありません。")
            st.rerun()
        waiting = sum(r["triage"] is None for r in rows)
        has_key = bool(os.getenv("ANTHROPIC_API_KEY"))
        if st.button(
            f"AIで仕分ける（未処理 {waiting}通）",
            type="primary",
            width="stretch",
            disabled=not waiting or not has_key,
        ):
            run_triage(store)
            st.rerun()
        if not has_key:
            st.info(".env に ANTHROPIC_API_KEY を設定すると仕分けできます。")
        st.divider()
        usage = store.usage_total()
        if usage:
            st.caption(
                f"入力 {usage.get('input_tokens', 0):,} ／ キャッシュ読み込み {usage.get('cache_read_tokens', 0):,} ／ "
                f"出力 {usage.get('output_tokens', 0):,} トークン"
            )
        st.caption(f"モデル：{MODEL}（effort: {EFFORT}）")


def detail(store: Store, row: dict) -> None:
    triage = row["triage"]
    left, right = st.columns(2, gap="large")
    with left:
        st.subheader(row["subject"])
        received = datetime.fromisoformat(row["received_at"]).strftime("%Y-%m-%d %H:%M")
        st.caption(f"{row['from_name']} <{row['from_addr']}>　{received}")
        with st.container(border=True):
            st.text(row["body"])
    with right:
        if triage is None:
            st.info("まだ仕分けていません。")
            return
        st.markdown(
            f"**{PRIORITY_LABEL[triage.priority]}**　{triage.category}　／　担当：{triage.assignee}　／　{triage.product}"
        )
        if triage.safety_risk:
            st.error("安全に関わるおそれがあります。最優先で対応してください。", icon="⚠️")
        st.caption(f"緊急度の理由：{triage.priority_reason}")
        st.markdown(f"**要点**　{triage.summary}")
        if triage.requests:
            st.markdown("**求めていること**\n" + "\n".join(f"- {r}" for r in triage.requests))
        if triage.missing_info:
            st.markdown("**確認が必要な情報**\n" + "\n".join(f"- {m}" for m in triage.missing_info))
        if triage.assignee == "対応不要":
            return
        reply = st.text_area("返信の下書き", value=row["reply"], height=320, key=f"reply-{row['id']}")
        if triage.reply_basis:
            st.caption("根拠：" + "／".join(triage.reply_basis))
        save, finish = st.columns(2)
        if save.button("下書きを保存", width="stretch", key=f"save-{row['id']}"):
            store.update_reply(row["id"], reply, done=False)
            st.toast("保存しました。")
        if finish.button("対応済みにする", type="primary", width="stretch", key=f"done-{row['id']}"):
            store.update_reply(row["id"], reply, done=True)
            st.rerun()
        st.caption("このアプリはメールを送信しません。送信は担当者がメールソフトから行います。")


def inbox_tab(store: Store, rows: list[dict]) -> None:
    status_col, priority_col, category_col = st.columns(3)
    statuses = status_col.multiselect(
        "状態", list(STATUSES), default=["new", "open"], format_func=STATUSES.get
    )
    priorities = priority_col.multiselect("緊急度", ["高", "中", "低"])
    categories = category_col.multiselect(
        "分類", sorted({r["triage"].category for r in rows if r["triage"]})
    )
    shown = [
        r
        for r in rows
        if r["status"] in statuses
        and (not priorities or (r["triage"] and r["triage"].priority in priorities))
        and (not categories or (r["triage"] and r["triage"].category in categories))
    ]
    shown.sort(key=lambda r: (PRIORITY_ORDER[r["triage"].priority] if r["triage"] else -1, r["received_at"]))
    table = pd.DataFrame(
        [
            {
                "緊急度": PRIORITY_LABEL[r["triage"].priority] if r["triage"] else "―",
                "分類": r["triage"].category if r["triage"] else "未処理",
                "件名": r["subject"],
                "差出人": r["from_name"],
                "担当": r["triage"].assignee if r["triage"] else "",
                "受信": datetime.fromisoformat(r["received_at"]).strftime("%m/%d %H:%M"),
                "状態": STATUSES[r["status"]],
            }
            for r in shown
        ]
    )
    if table.empty:
        st.write("該当するメールはありません。左の「受信箱を取り込む」から始めてください。")
        return
    event = st.dataframe(
        table, hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row", key="inbox"
    )
    selected = event.selection.rows
    st.divider()
    if selected:
        detail(store, shown[selected[0]])
    else:
        st.caption("一覧の行を選ぶと、メール本文と仕分け結果、返信の下書きが表示されます。")


def summary_tab(rows: list[dict]) -> None:
    triaged = [r["triage"] for r in rows if r["triage"]]
    if not triaged:
        st.write("仕分けたメールがありません。")
        return
    frame = pd.DataFrame([t.model_dump() for t in triaged])
    counts = frame["priority"].value_counts()
    high, middle, low = st.columns(3)
    high.metric(PRIORITY_LABEL["高"], int(counts.get("高", 0)))
    middle.metric(PRIORITY_LABEL["中"], int(counts.get("中", 0)))
    low.metric(PRIORITY_LABEL["低"], int(counts.get("低", 0)))
    left, right = st.columns(2)
    for column, field, title in ((left, "category", "分類別"), (right, "assignee", "担当別")):
        column.markdown(f"**{title}**")
        data = frame[field].value_counts().rename_axis(title).reset_index(name="件数")
        column.bar_chart(data, x=title, y="件数", horizontal=True, sort="-件数", x_label="件数", y_label="", height=320)


def export_tab(rows: list[dict]) -> None:
    records = [
        {
            "受信日時": r["received_at"],
            "差出人": r["from_name"],
            "メールアドレス": r["from_addr"],
            "件名": r["subject"],
            "状態": STATUSES[r["status"]],
            "分類": r["triage"].category if r["triage"] else "",
            "緊急度": r["triage"].priority if r["triage"] else "",
            "担当": r["triage"].assignee if r["triage"] else "",
            "製品": r["triage"].product if r["triage"] else "",
            "要点": r["triage"].summary if r["triage"] else "",
            "返信": r["reply"],
        }
        for r in rows
    ]
    st.caption("仕分け結果をCSVで書き出します。Excelやスプレッドシートでそのまま開けます。")
    st.download_button(
        "CSVを書き出す",
        pd.DataFrame(records).to_csv(index=False).encode("utf-8-sig"),
        file_name="inquiries.csv",
        mime="text/csv",
        disabled=not records,
    )


def main() -> None:
    st.set_page_config(page_title="問い合わせ仕分け（デモ）", page_icon="📨", layout="wide")
    store = get_store()
    rows = store.all()
    sidebar(store, rows)

    st.header("ギシ電機 サポート窓口 受信箱")
    triaged = [r for r in rows if r["triage"]]
    a, b, c, d = st.columns(4)
    a.metric("要対応", sum(r["status"] == "open" for r in rows))
    b.metric("うち緊急度「高」", sum(r["status"] == "open" and r["triage"].priority == "高" for r in triaged))
    c.metric("対応済み", sum(r["status"] == "done" for r in rows))
    d.metric("未処理", sum(r["triage"] is None for r in rows))

    inbox, summary, export = st.tabs(["受信箱", "集計", "書き出し"])
    with inbox:
        inbox_tab(store, rows)
    with summary:
        summary_tab(rows)
    with export:
        export_tab(rows)


main()
