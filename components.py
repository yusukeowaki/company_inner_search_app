"""
このファイルは、画面表示に特化した関数定義のファイルです。
"""

############################################################
# ライブラリの読み込み
############################################################
import streamlit as st
import utils
import constants as ct

# ===== ここから追加：共通ラベル関数 =====
def label_with_page(path: str, page) -> str:
    """表示用ラベル: パスの区切りを'/'に正規化 + ページがあれば『（ページNo.n）』を付与"""
    disp = (path or "").replace("\\", "/")
    if page is None:
        return disp
    try:
        n = int(page) + 1  # 0始まりを1始まりに
    except Exception:
        return f"{disp} （ページNo.{page}）"
    return f"{disp} （ページNo.{n}）"
# ===== ここまで追加 =====

############################################################
# 関数定義
############################################################

def display_app_title():
    """
    タイトル表示
    """
    st.markdown(f"## {ct.APP_NAME}")


def display_select_mode():
    """回答モードのラジオボタンをサイドバーに表示"""
    st.sidebar.header("利用目的")

    # すでに選択済みならそれを初期値にする
    current = st.session_state.get("mode", ct.ANSWER_MODE_1)
    options = (ct.ANSWER_MODE_1, ct.ANSWER_MODE_2)
    index = 0 if current == ct.ANSWER_MODE_1 else 1

    st.session_state.mode = st.sidebar.radio(
        "モードを選択",                 # ← 空ラベルにしない（警告回避）
        options=options,
        index=index,
        help="用途に応じて切り替えてください",
        key="mode_radio",
    )
    
    st.sidebar.markdown("**【「社内文書検索」を選択した場合】**")
    st.sidebar.info("入力内容と関連性が高い社内文書のありかを検索できます。")
    st.sidebar.markdown("**【入力例】**")
    st.sidebar.code("社員の育成方針に関するMTGの議事録", wrap_lines=True, language=None)

    st.sidebar.markdown("**【「社内問い合わせ」を選択した場合】**")
    st.sidebar.info("質問・要望に対して、社内文書の情報をもとに回答を得られます。")
    st.sidebar.markdown("**【入力例】**")
    st.sidebar.code("人事部に所属している従業員情報を一覧化して", wrap_lines=True, language=None)

 
    
    
    # ← ここから追加（モード選択の直下に表示）
    st.sidebar.divider()
    if st.sidebar.button("履歴クリア", type="secondary", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

def display_initial_ai_message():
    """
    AIメッセージの初期表示
    """
    with st.chat_message("assistant"):
        # 「st.success()」とすると緑枠で表示される
        st.success("こんにちは。私は社内文書の情報をもとに回答する生成AIチャットボットです。上記で利用目的を選択し、画面下部のチャット欄からメッセージを送信してください。")
         # ★ 黄色の注意（ここを追加）
        st.warning("具体的に入力したほうが精度が高めの回答を得やすいです。")
        
        # 「社内文書検索」の機能説明
        #st.markdown("**【「社内文書検索」を選択した場合】**")
        # 「st.info()」を使うと青枠で表示される
        #st.info("入力内容と関連性が高い社内文書のありかを検索できます。")
        # 「st.code()」を使うとコードブロックの装飾で表示される
        # 「wrap_lines=True」で折り返し設定、「language=None」で非装飾とする
        #st.code("【入力例】\n社員の育成方針に関するMTGの議事録", wrap_lines=True, language=None)

        # 「社内問い合わせ」の機能説明
        #st.markdown("**【「社内問い合わせ」を選択した場合】**")
        #st.info("質問・要望に対して、社内文書の情報をもとに回答を得られます。")
        #st.code("【入力例】\n人事部に所属している従業員情報を一覧化して", wrap_lines=True, language=None)


def display_conversation_log():
    """
    会話ログの一覧表示
    """
    # 会話ログのループ処理
    for message in st.session_state.messages:
        # 「message」辞書の中の「role」キーには「user」か「assistant」が入っている
        with st.chat_message(message["role"]):

            # ユーザー入力値の場合、そのままテキストを表示するだけ
            if message["role"] == "user":
                st.markdown(message["content"])
            
            # LLMからの回答の場合
            else:
                # 「社内文書検索」の場合、テキストの種類に応じて表示形式を分岐処理
                if message["content"]["mode"] == ct.ANSWER_MODE_1:
                    
                    # ファイルのありかの情報が取得できた場合（通常時）の表示処理
                    if not "no_file_path_flg" in message["content"]:
                        # ==========================================
                        # ユーザー入力値と最も関連性が高いメインドキュメントのありかを表示
                        # ==========================================
                        # 補足文の表示
                        st.markdown(message["content"]["main_message"])

                        # 参照元のアイコン
                        icon = utils.get_source_icon(message["content"]["main_file_path"])

                        # ページ番号（0始まり or None）を安全に取得
                        page = message["content"].get("main_page_number")

                        # ラベルを組み立て（ページが取れた時だけ p.◯ を付ける）
                        label = label_with_page(message["content"]["main_file_path"], page)

                        # 表示
                        st.success(label, icon=icon)
                        # ==========================================
                        # ユーザー入力値と関連性が高いサブドキュメントのありかを表示
                        # ==========================================
                        if "sub_message" in message["content"]:
                            # 補足メッセージの表示
                            st.markdown(message["content"]["sub_message"])

                            # サブドキュメントのありかを一覧表示
                            for sub_choice in message["content"]["sub_choices"]:
                                # アイコン
                                icon = utils.get_source_icon(sub_choice["source"])
                                
                                # 0始まりのページ番号（ない時は None）
                                page = sub_choice.get("page_number")

                                # ページが取れた時だけ p.◯ を付ける
                                label = label_with_page(sub_choice["source"], page)

                                st.info(label, icon=icon)
                    
                    
                    # ファイルのありかの情報が取得できなかった場合、LLMからの回答のみ表示
                    else:
                        st.markdown(message["content"]["answer"])
                
                # 「社内問い合わせ」の場合の表示処理
                else:
                    # LLMからの回答を表示
                    st.markdown(message["content"]["answer"])

                    # 参照元のありかを一覧表示
                    if "file_info_list" in message["content"]:
                        # 区切り線の表示
                        st.divider()
                        # 「情報源」の文字を太字で表示
                        st.markdown(f"##### {message['content']['message']}")
                        # ドキュメントのありかを一覧表示
                        for item in message["content"]["file_info_list"]:
                        # 新旧どちらの形式でも動くように吸収
                            if isinstance(item, dict):
                                label  = item.get("label", "")
                                source = item.get("source", "")
                            else:
                        # 互換: 過去の履歴(文字列のみ)でも表示できるように
                                label  = item
                                source = item
                            
                            icon = utils.get_source_icon(source)
                            st.info(label, icon=icon)


def display_search_llm_response(llm_response):
    """
    「社内文書検索」モードにおけるLLMレスポンスを表示

    Args:
        llm_response: LLMからの回答

    Returns:
        LLMからの回答を画面表示用に整形した辞書データ
    """
    # LLMからのレスポンスに参照元情報が入っており、かつ「該当資料なし」が回答として返された場合
    if llm_response["context"] and llm_response["answer"] != ct.NO_DOC_MATCH_ANSWER:

        # ==========================================
        # ユーザー入力値と最も関連性が高いメインドキュメントのありかを表示
        # ==========================================
        # LLMからのレスポンス（辞書）の「context」属性の中の「0」に、最も関連性が高いドキュメント情報が入っている
        main_file_path = llm_response["context"][0].metadata["source"]

        # 補足メッセージの表示
        main_message = "入力内容に関する情報は、以下のファイルに含まれている可能性があります。"
        st.markdown(main_message)
        
        # 参照元のありかに応じて、適したアイコンを取得
        icon = utils.get_source_icon(main_file_path)
        
        # ページ番号が取得できた場合のみ、ページ番号を表示（ドキュメントによっては取得できない場合がある）
        page = llm_response["context"][0].metadata.get("page") 
        label = label_with_page(main_file_path, page)
        st.success(label, icon=icon)

        # ==========================================
        # ユーザー入力値と関連性が高いサブドキュメントのありかを表示
        # ==========================================
        # メインドキュメント以外で、関連性が高いサブドキュメントを格納する用のリストを用意
        sub_choices = []
        # 重複チェック用のリストを用意
        duplicate_check_list = []

        # ドキュメントが2件以上検索できた場合（サブドキュメントが存在する場合）のみ、サブドキュメントのありかを一覧表示
        # 「source_documents」内のリストの2番目以降をスライスで参照（2番目以降がなければfor文内の処理は実行されない）
        for document in llm_response["context"][1:]:
            # ドキュメントのファイルパスを取得
            sub_file_path = document.metadata["source"]

            # メインドキュメントのファイルパスと重複している場合、処理をスキップ（表示しない）
            if sub_file_path == main_file_path:
                continue
            
            # 同じファイル内の異なる箇所を参照した場合、2件目以降のファイルパスに重複が発生する可能性があるため、重複を除去
            if sub_file_path in duplicate_check_list:
                continue

            # 重複チェック用のリストにファイルパスを順次追加
            duplicate_check_list.append(sub_file_path)
            
            # ページ番号が取得できない場合のための分岐処理
            if "page" in document.metadata:
                # ページ番号を取得
                sub_page_number = document.metadata["page"]
                # 「サブドキュメントのファイルパス」と「ページ番号」の辞書を作成
                sub_choice = {"source": sub_file_path, "page_number": sub_page_number}
            else:
                # 「サブドキュメントのファイルパス」の辞書を作成
                sub_choice = {"source": sub_file_path}
            
            # 後ほど一覧表示するため、サブドキュメントに関する情報を順次リストに追加
            sub_choices.append(sub_choice)
        
        # サブドキュメントが存在する場合のみの処理
        if sub_choices:
            # 補足メッセージの表示
            sub_message = "その他、ファイルありかの候補を提示します。"
            st.markdown(sub_message)

            # サブドキュメントに対してのループ処理
            for sub_choice in sub_choices:
                # 参照元のありかに応じて、適したアイコンを取得
                icon = utils.get_source_icon(sub_choice['source'])
                # ページ番号（0始まり）を安全に取得
                page = sub_choice.get("page_number")

                # ラベルを組み立て（ページが取れた時だけ p.◯ を付ける）
                label = label_with_page(sub_choice["source"], page)

                st.info(label, icon=icon)
                
                
                
        
        # 表示用の会話ログに格納するためのデータを用意
        # - 「mode」: モード（「社内文書検索」or「社内問い合わせ」）
        # - 「main_message」: メインドキュメントの補足メッセージ
        # - 「main_file_path」: メインドキュメントのファイルパス
        # - 「main_page_number」: メインドキュメントのページ番号
        # - 「sub_message」: サブドキュメントの補足メッセージ
        # - 「sub_choices」: サブドキュメントの情報リスト
        content = {}
        content["mode"] = ct.ANSWER_MODE_1
        content["main_message"] = main_message
        content["main_file_path"] = main_file_path
        # メインドキュメントのページ番号は、取得できた場合のみ追加
        page = llm_response["context"][0].metadata.get("page")  # 0始まり or None
        if page is not None:
            content["main_page_number"] = page  # 保持は0始まりのまま（表示時に +1）
        # サブドキュメントの情報は、取得できた場合にのみ追加
        if sub_choices:
            content["sub_message"] = sub_message
            content["sub_choices"] = sub_choices
    
    # LLMからのレスポンスに、ユーザー入力値と関連性の高いドキュメント情報が入って「いない」場合
    else:
        # 関連ドキュメントが取得できなかった場合のメッセージ表示
        st.markdown(ct.NO_DOC_MATCH_MESSAGE)

        # 表示用の会話ログに格納するためのデータを用意
        # - 「mode」: モード（「社内文書検索」or「社内問い合わせ」）
        # - 「answer」: LLMからの回答
        # - 「no_file_path_flg」: ファイルパスが取得できなかったことを示すフラグ（画面を再描画時の分岐に使用）
        content = {}
        content["mode"] = ct.ANSWER_MODE_1
        content["answer"] = ct.NO_DOC_MATCH_MESSAGE
        content["no_file_path_flg"] = True
    
    return content


def display_contact_llm_response(llm_response):
    """
    「社内問い合わせ」モードにおけるLLMレスポンスを表示

    Args:
        llm_response: LLMからの回答

    Returns:
        LLMからの回答を画面表示用に整形した辞書データ
    """
    # LLMからの回答を表示
    st.markdown(llm_response["answer"])

    # ユーザーの質問・要望に適切な回答を行うための情報が、社内文書のデータベースに存在しなかった場合
    if llm_response["answer"] != ct.INQUIRY_NO_MATCH_ANSWER:
        # 区切り線を表示
        st.divider()

        # 補足メッセージを表示
        message = "情報源"
        st.markdown(f"##### {message}")

        # 参照元のファイルパスの一覧を格納するためのリストを用意
        file_path_list = []
        file_info_list = []

        # LLMが回答生成の参照元として使ったドキュメントの一覧が「context」内のリストの中に入っているため、ループ処理
        for document in llm_response["context"]:
            # ファイルパスを取得
            file_path = document.metadata["source"]
            # ファイルパスの重複は除去
            if file_path in file_path_list:
                continue

            # ★ 0始まりのページ番号を安全に取得（無いときは None）
            page = document.metadata.get("page")

            # ★ ページがあるときだけ p.◯ を付けたラベルを作成
            label = label_with_page(file_path, page)

            # ★ アイコンはファイルパスから判定（ラベルではなくパスを渡す）
            icon = utils.get_source_icon(file_path)
            st.info(label, icon=icon)

            # ★ 重複チェック用と表示用に保存
            file_path_list.append(file_path)
            file_info_list.append({"label": label, "source": file_path})
            
    # 表示用の会話ログに格納するためのデータを用意
    # - 「mode」: モード（「社内文書検索」or「社内問い合わせ」）
    # - 「answer」: LLMからの回答
    # - 「message」: 補足メッセージ
    # - 「file_path_list」: ファイルパスの一覧リスト
    content = {}
    content["mode"] = ct.ANSWER_MODE_2
    content["answer"] = llm_response["answer"]
    # 参照元のドキュメントが取得できた場合のみ
    if llm_response["answer"] != ct.INQUIRY_NO_MATCH_ANSWER:
        content["message"] = message
        content["file_info_list"] = file_info_list

    return content