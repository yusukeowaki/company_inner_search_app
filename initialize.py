import os
import sys
import logging
from logging.handlers import TimedRotatingFileHandler
from uuid import uuid4
import unicodedata

import streamlit as st
from dotenv import load_dotenv

# ---- 基本設定（インポート時に UI は呼ばない）----
os.environ.setdefault("USER_AGENT", "company_inner_search_app/1.0")
load_dotenv()  # ローカル .env

def get_env(name, default=None):
    return os.getenv(name) or st.secrets.get(name, default)

OPENAI_API_KEY = get_env("OPENAI_API_KEY")
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY  # langchain/openai が参照
# 未設定でもここでは UI を出さない。initialize() 内でガードする。

# --- SQLite(FTS5) patch for Chroma ---
try:
    import pysqlite3  # requirements: pysqlite3-binary
    sys.modules["sqlite3"] = pysqlite3
except Exception:
    pass

# ---- 以降の依存 ----
import constants as ct
import utils
# from docx import Document  # 未使用ならコメントアウト
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document as LCDocument
import pandas as pd

# ---------------------------
# 初期化エントリポイント
# ---------------------------
def initialize():
    """画面読み込み時に実行する初期化処理（UI呼び出しはこの関数内だけ）"""
    initialize_session_state()
    initialize_session_id()
    initialize_logger()

    # ★ APIキー未設定の案内はここで（UI可）
    if not get_env("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY が未設定です。Streamlit Cloud の Secrets に追加してください。")
        return

    initialize_retriever()


# ---------------------------
# ロガー
# ---------------------------
def initialize_logger():
    os.makedirs(ct.LOG_DIR_PATH, exist_ok=True)
    logger = logging.getLogger(ct.LOGGER_NAME)
    if logger.hasHandlers():
        return

    log_handler = TimedRotatingFileHandler(
        os.path.join(ct.LOG_DIR_PATH, ct.LOG_FILE),
        when="D",
        encoding="utf8"
    )
    formatter = logging.Formatter(
        f"[%(levelname)s] %(asctime)s line %(lineno)s, in %(funcName)s, session_id={st.session_state.session_id}: %(message)s"
    )
    log_handler.setFormatter(formatter)
    logger.setLevel(logging.INFO)
    logger.addHandler(log_handler)


# ---------------------------
# セッション
# ---------------------------
def initialize_session_id():
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid4().hex

def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.chat_history = []


# ---------------------------
# Retriever 構築
# ---------------------------
def initialize_retriever():
    logger = logging.getLogger(ct.LOGGER_NAME)
    if "retriever" in st.session_state:
        return

    docs_all = load_data_sources()

    # 文字列調整（Windows対応）
    for doc in docs_all:
        doc.page_content = adjust_string(doc.page_content)
        for key in list(doc.metadata.keys()):
            doc.metadata[key] = adjust_string(doc.metadata[key])

    # ★ APIキー依存の処理なのでここで最終確認
    if not get_env("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY 未設定")

    embeddings = OpenAIEmbeddings()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=ct.CHUNK_SIZE,
        chunk_overlap=ct.CHUNK_OVERLAP,
    )

    def _truthy(v):
        s = str(v).strip().lower()
        return v is True or s in {"true", "1", "yes", "y"}

    no_split_docs = [d for d in docs_all if _truthy(d.metadata.get("no_split"))]
    to_split_docs = [d for d in docs_all if not _truthy(d.metadata.get("no_split"))]

    splitted_docs = no_split_docs + text_splitter.split_documents(to_split_docs)

    db = Chroma.from_documents(
        splitted_docs,
        embedding=embeddings,
        persist_directory=getattr(ct, "CHROMA_DIR", "chroma_db"),
    )
    st.session_state.retriever = db.as_retriever(search_kwargs={"k": ct.RETRIEVE_TOP_K})


# ---------------------------
# データ読み込み
# ---------------------------
def load_data_sources():
    docs_all = []

    if os.path.exists(ct.RAG_TOP_FOLDER_PATH):
        recursive_file_check(ct.RAG_TOP_FOLDER_PATH, docs_all)
    else:
        logging.getLogger(ct.LOGGER_NAME).warning(
            f"RAG_TOP_FOLDER_PATH が存在しません: {ct.RAG_TOP_FOLDER_PATH}"
        )

    web_docs_all = []
    for web_url in ct.WEB_URL_LOAD_TARGETS:
        try:
            loader = WebBaseLoader(web_url)
            web_docs = loader.load()
            web_docs_all.extend(web_docs)
        except Exception as e:
            logging.getLogger(ct.LOGGER_NAME).warning(f"Web 読み込み失敗: {web_url} ({e})")
    docs_all.extend(web_docs_all)
    return docs_all


def recursive_file_check(path, docs_all):
    if os.path.isdir(path):
        for name in os.listdir(path):
            full = os.path.join(path, name)
            recursive_file_check(full, docs_all)
    else:
        file_load(path, docs_all)


def file_load(path, docs_all):
    file_extension = os.path.splitext(path)[1]
    file_name = os.path.basename(path)

    # ★ 社員名簿.csv は 1ドキュメント統合
    if file_extension.lower() == ".csv" and file_name == "社員名簿.csv":
        _load_and_merge_employee_csv(path, docs_all)
        return

    if file_extension in ct.SUPPORTED_EXTENSIONS:
        loader = ct.SUPPORTED_EXTENSIONS[file_extension](path)
        docs = loader.load()
        docs_all.extend(docs)


def _row_to_line(row: dict, expected_cols: list[str]) -> str:
    lines = []
    aliases = {
        "氏名（フルネーム）": ["氏名（フルネーム）", "氏名", "フルネーム"],
        "メールアドレス": ["メールアドレス", "メール", "mail", "e-mail"],
        "学部・学科": ["学部・学科", "学部/学科", "学部学科"],
    }
    for col in expected_cols:
        keys = [col] + aliases.get(col, [])
        val = ""
        for k in keys:
            if k in row and str(row[k]).strip():
                val = str(row[k]).strip()
                break
        if val and val.lower() != "nan":
            lines.append(f"{col}:{val}")
    return "\n".join(lines)


def _load_and_merge_employee_csv(path: str, docs_all: list) -> None:
    logger = logging.getLogger(ct.LOGGER_NAME)

    df = None
    last_err = None
    for enc in ("utf-8-sig", "utf-8", "cp932"):
        try:
            df = pd.read_csv(path, encoding=enc).fillna("")
            break
        except Exception as e:
            last_err = e
    if df is None:
        logger.warning(f"社員名簿の読み込みに失敗しました: {path} ({last_err})")
        return

    expected_cols = [
        "社員ID", "氏名（フルネーム）", "性別", "生年月日", "年齢",
        "メールアドレス", "従業員区分", "入社日", "役職",
        "スキルセット", "保有資格", "大学名", "学部・学科", "卒業年月日",
        "部署",
    ]
    present_cols = [c for c in expected_cols if c in df.columns or c in ["氏名（フルネーム）","メールアドレス","学部・学科"]]

    if "部署" in df.columns:
        for dept, sub in df.groupby("部署", dropna=False):
            dept_name = (str(dept).strip() or "（部署不明）")
            records = [_row_to_line(row, present_cols) for _, row in sub.iterrows()]
            content = f"社員名簿｜部署:{dept_name}\n" + "\n\n".join(records)
            docs_all.append(
                LCDocument(page_content=content, metadata={"source": path, "dept": dept_name, "no_split": True})
            )
    else:
        records = [_row_to_line(row, present_cols) for _, row in df.iterrows()]
        content = "社員名簿（部署列なし）\n" + "\n\n".join(records)
        docs_all.append(
            LCDocument(page_content=content, metadata={"source": path, "no_split": True})
        )


def adjust_string(s):
    if type(s) is not str:
        return s
    if sys.platform.startswith("win"):
        s = unicodedata.normalize('NFC', s)
        s = s.encode("cp932", "ignore").decode("cp932")
        return s
    return s