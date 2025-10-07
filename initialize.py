import os
os.environ.setdefault("USER_AGENT", "company_inner_search_app/1.0")

# .env → st.secrets の順で読み取る
import os, streamlit as st
from dotenv import load_dotenv
load_dotenv()

def get_env(name, default=None):
    return os.getenv(name) or st.secrets.get(name, default)

OPENAI_API_KEY = get_env("OPENAI_API_KEY")
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY  # langchain/openai が参照

# ない場合は画面に分かりやすくエラー表示
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY が未設定です。Streamlit Cloud の Secrets に追加してください。")


"""
このファイルは、最初の画面読み込み時にのみ実行される初期化処理が記述されたファイルです。
"""
# --- SQLite(FTS5) patch for Chroma ---
try:
    import sys, pysqlite3  # requirements.txt に pysqlite3-binary を入れておく
    sys.modules["sqlite3"] = pysqlite3
except Exception:
    pass
# -------------------------------------
############################################################
# ライブラリの読み込み
############################################################
import os
import logging
from logging.handlers import TimedRotatingFileHandler
from uuid import uuid4
import sys
import unicodedata
from dotenv import load_dotenv
import streamlit as st

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import constants as ct
import utils
import pandas as pd
from langchain_core.documents import Document as LCDocument
############################################################
# 設定関連
############################################################
# 「.env」ファイルで定義した環境変数の読み込み
load_dotenv()


############################################################
# 関数定義
############################################################
def _row_to_line(row: dict, expected_cols: list[str]) -> str:
    """1人分を課題の列順で '列名:値' の縦並びに整形"""
    lines = []
    for col in expected_cols:
        # CSV列名の揺れ吸収（例: 氏名/氏名（フルネーム）、メール/メールアドレス など）
        aliases = {
            "氏名（フルネーム）": ["氏名（フルネーム）", "氏名", "フルネーム"],
            "メールアドレス": ["メールアドレス", "メール", "mail", "e-mail"],
            "学部・学科": ["学部・学科", "学部/学科", "学部学科"],
        }
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
    """社員名簿.csvを『部署ごとに1ドキュメント』にまとめ、分割しないフラグ付きで追加"""
    logger = logging.getLogger(ct.LOGGER_NAME)

    import pandas as pd
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

    # 課題の列順（無い列はスキップ）
    expected_cols = [
        "社員ID", "氏名（フルネーム）", "性別", "生年月日", "年齢",
        "メールアドレス", "従業員区分", "入社日", "役職",
        "スキルセット", "保有資格", "大学名", "学部・学科", "卒業年月日",
        # 検索条件として使うので部署も保持
        "部署",
    ]
    present_cols = [c for c in expected_cols if c in df.columns or c in ["氏名（フルネーム）","メールアドレス","学部・学科"]]

    if "部署" in df.columns:
        for dept, sub in df.groupby("部署", dropna=False):
            dept_name = (str(dept).strip() or "（部署不明）")
            records = [_row_to_line(row, present_cols) for _, row in sub.iterrows()]
            # 1ドキュメント＝1部署、各人は空行区切りで並べる
            content = f"社員名簿｜部署:{dept_name}\n" + "\n\n".join(records)
            docs_all.append(
                LCDocument(
                    page_content=content,
                    metadata={"source": path, "dept": dept_name, "no_split": True},
                )
            )
    else:
        records = [_row_to_line(row, present_cols) for _, row in df.iterrows()]
        content = "社員名簿（部署列なし）\n" + "\n\n".join(records)
        docs_all.append(
            LCDocument(
                page_content=content,
                metadata={"source": path, "no_split": True},
            )
        )


def initialize():
    """
    画面読み込み時に実行する初期化処理
    """
    # 初期化データの用意
    initialize_session_state()
    # ログ出力用にセッションIDを生成
    initialize_session_id()
    # ログ出力の設定
    initialize_logger()
    # RAGのRetrieverを作成
    initialize_retriever()


def initialize_logger():
    """
    ログ出力の設定
    """
    # 指定のログフォルダが存在すれば読み込み、存在しなければ新規作成
    os.makedirs(ct.LOG_DIR_PATH, exist_ok=True)
    
    # 引数に指定した名前のロガー（ログを記録するオブジェクト）を取得
    # 再度別の箇所で呼び出した場合、すでに同じ名前のロガーが存在していれば読み込む
    logger = logging.getLogger(ct.LOGGER_NAME)

    # すでにロガーにハンドラー（ログの出力先を制御するもの）が設定されている場合、同じログ出力が複数回行われないよう処理を中断する
    if logger.hasHandlers():
        return

    # 1日単位でログファイルの中身をリセットし、切り替える設定
    log_handler = TimedRotatingFileHandler(
        os.path.join(ct.LOG_DIR_PATH, ct.LOG_FILE),
        when="D",
        encoding="utf8"
    )
    # 出力するログメッセージのフォーマット定義
    # - 「levelname」: ログの重要度（INFO, WARNING, ERRORなど）
    # - 「asctime」: ログのタイムスタンプ（いつ記録されたか）
    # - 「lineno」: ログが出力されたファイルの行番号
    # - 「funcName」: ログが出力された関数名
    # - 「session_id」: セッションID（誰のアプリ操作か分かるように）
    # - 「message」: ログメッセージ
    formatter = logging.Formatter(
        f"[%(levelname)s] %(asctime)s line %(lineno)s, in %(funcName)s, session_id={st.session_state.session_id}: %(message)s"
    )

    # 定義したフォーマッターの適用
    log_handler.setFormatter(formatter)

    # ログレベルを「INFO」に設定
    logger.setLevel(logging.INFO)

    # 作成したハンドラー（ログ出力先を制御するオブジェクト）を、
    # ロガー（ログメッセージを実際に生成するオブジェクト）に追加してログ出力の最終設定
    logger.addHandler(log_handler)


def initialize_session_id():
    """
    セッションIDの作成
    """
    if "session_id" not in st.session_state:
        # ランダムな文字列（セッションID）を、ログ出力用に作成
        st.session_state.session_id = uuid4().hex


def initialize_retriever():
    """
    画面読み込み時にRAGのRetriever（ベクターストアから検索するオブジェクト）を作成
    """
    # ロガーを読み込むことで、後続の処理中に発生したエラーなどがログファイルに記録される
    logger = logging.getLogger(ct.LOGGER_NAME)

    # すでにRetrieverが作成済みの場合、後続の処理を中断
    if "retriever" in st.session_state:
        return
    
    # RAGの参照先となるデータソースの読み込み
    docs_all = load_data_sources()

    # OSがWindowsの場合、Unicode正規化と、cp932（Windows用の文字コード）で表現できない文字を除去
    for doc in docs_all:
        doc.page_content = adjust_string(doc.page_content)
        for key in doc.metadata:
            doc.metadata[key] = adjust_string(doc.metadata[key])
    
    # 埋め込みモデルの用意
    embeddings = OpenAIEmbeddings()
    
    # チャンク分割用のオブジェクトを作成
   
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=ct.CHUNK_SIZE,
    chunk_overlap=ct.CHUNK_OVERLAP,
    )

    # チャンク分割を実施（no_split フラグを尊重）

    def _truthy(v):
        s = str(v).strip().lower()
        return v is True or s in {"true", "1", "yes", "y"}
    
    no_split_docs = [d for d in docs_all if _truthy(d.metadata.get("no_split"))]
    to_split_docs = [d for d in docs_all if not _truthy(d.metadata.get("no_split"))]

    splitted_docs = no_split_docs + text_splitter.split_documents(to_split_docs)
    # ベクターストアの作成
    db = Chroma.from_documents(splitted_docs, embedding=embeddings)

    # ベクターストアを検索するRetrieverの作成
    st.session_state.retriever = db.as_retriever(search_kwargs={"k": ct.RETRIEVE_TOP_K})


def initialize_session_state():
    """
    初期化データの用意
    """
    if "messages" not in st.session_state:
        # 「表示用」の会話ログを順次格納するリストを用意
        st.session_state.messages = []
        # 「LLMとのやりとり用」の会話ログを順次格納するリストを用意
        st.session_state.chat_history = []


def load_data_sources():
    """
    RAGの参照先となるデータソースの読み込み

    Returns:
        読み込んだ通常データソース
    """
    # データソースを格納する用のリスト
    docs_all = []
    # ファイル読み込みの実行（渡した各リストにデータが格納される）
    recursive_file_check(ct.RAG_TOP_FOLDER_PATH, docs_all)

    web_docs_all = []
    # ファイルとは別に、指定のWebページ内のデータも読み込み
    # 読み込み対象のWebページ一覧に対して処理
    for web_url in ct.WEB_URL_LOAD_TARGETS:
        # 指定のWebページを読み込み
        loader = WebBaseLoader(web_url)
        web_docs = loader.load()
        # for文の外のリストに読み込んだデータソースを追加
        web_docs_all.extend(web_docs)
    # 通常読み込みのデータソースにWebページのデータを追加
    docs_all.extend(web_docs_all)

    return docs_all


def recursive_file_check(path, docs_all):
    """
    RAGの参照先となるデータソースの読み込み

    Args:
        path: 読み込み対象のファイル/フォルダのパス
        docs_all: データソースを格納する用のリスト
    """
    # パスがフォルダかどうかを確認
    if os.path.isdir(path):
        # フォルダの場合、フォルダ内のファイル/フォルダ名の一覧を取得
        files = os.listdir(path)
        # 各ファイル/フォルダに対して処理
        for file in files:
            # ファイル/フォルダ名だけでなく、フルパスを取得
            full_path = os.path.join(path, file)
            # フルパスを渡し、再帰的にファイル読み込みの関数を実行
            recursive_file_check(full_path, docs_all)
    else:
        # パスがファイルの場合、ファイル読み込み
        file_load(path, docs_all)


def file_load(path, docs_all):
    """
    ファイル内のデータ読み込み

    Args:
        path: ファイルパス
        docs_all: データソースを格納する用のリスト
    """
    # ファイルの拡張子を取得
    file_extension = os.path.splitext(path)[1]
    # ファイル名（拡張子を含む）を取得
    file_name = os.path.basename(path)
    
     # ★ 社員名簿.csv は行分割せず 1 ドキュメントに統合して読み込む
    if file_extension.lower() == ".csv" and file_name == "社員名簿.csv":
        _load_and_merge_employee_csv(path, docs_all)
        return
    # ★ ここまで

    # 想定していたファイル形式の場合のみ読み込む
    if file_extension in ct.SUPPORTED_EXTENSIONS:
        # ファイルの拡張子に合ったdata loaderを使ってデータ読み込み
        loader = ct.SUPPORTED_EXTENSIONS[file_extension](path)
        docs = loader.load()
        docs_all.extend(docs)


def adjust_string(s):
    """
    Windows環境でRAGが正常動作するよう調整
    
    Args:
        s: 調整を行う文字列
    
    Returns:
        調整を行った文字列
    """
    # 調整対象は文字列のみ
    if type(s) is not str:
        return s

    # OSがWindowsの場合、Unicode正規化と、cp932（Windows用の文字コード）で表現できない文字を除去
    if sys.platform.startswith("win"):
        s = unicodedata.normalize('NFC', s)
        s = s.encode("cp932", "ignore").decode("cp932")
        return s
    
    # OSがWindows以外の場合はそのまま返す
    return s