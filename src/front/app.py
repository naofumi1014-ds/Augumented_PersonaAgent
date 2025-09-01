import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.agent.request import UserQuery
from openai import AzureOpenAI
import streamlit as st
import json
from datetime import datetime
from dotenv import load_dotenv
from src.agent.personalization_method import Personalization



# 環境変数を読み込み
load_dotenv()

# 環境変数から直接取得する場合の優先順位を設定
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
version = os.getenv("AZURE_OPENAI_API_VERSION")

# アカウント情報
if "account" not in st.session_state:
    st.session_state.account = None
if "chat_id" not in st.session_state:
    st.session_state.chat_id = datetime.now().strftime("%Y%m%d_%H%M%S")

USER = ["osawa","userA","userB","userC"]

def start_new_chat():
    """新しいチャットを開始する関数"""

    #ri_liの推論
    personalization.generate_episodic_memory(st.session_state.chat_id, st.session_state.user)

    # ! Episodic Memoryの取得 toolのテスト
    # personalization.get_episodic_memory(
    #         user_query=user_query(user_query="テストクエリ"),
    #         user=st.session_state.user,
    #         top_k=5
    #     )
    
    # 新しいチャットIDを生成
    st.session_state.chat_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # メッセージをリセット
    st.session_state.messages = []


# サイドバーにアカウント選択を追加
with st.sidebar:
    st.text("Account")
    st.session_state.user = st.selectbox("Choose your account", USER)
    personalization = Personalization(st.session_state.user)

    st.markdown("---")

    # 新規チャットボタン
    if st.button("会話を終了\n\n\n（新規チャットを生成）", use_container_width=True):
        start_new_chat()
        st.rerun()
    
    # 現在のチャットID表示
    st.text(f"Chat ID: {st.session_state.chat_id}")



st.title("PersonaAgent Chatbot")
st.caption("PersonaAgent検証用")
if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])



if prompt := st.chat_input():

    if not api_key or not endpoint or not deployment:
        st.info("Please add your Azure OpenAI API key, endpoint, and deployment name to continue.")
        st.stop()

    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=version
    )
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    try:
        response = client.chat.completions.create(
            model=deployment,
            messages=st.session_state.messages
        )
        msg = response.choices[0].message.content
        
        # レスポンスがNoneの場合のエラーハンドリング
        if msg is None:
            msg = "申し訳ございませんが、応答を生成できませんでした。"
        
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.chat_message("assistant").write(msg)
        
        # 会話を自動保存（メッセージが追加されるたびに）
        personalization.save_chat_log(
            chat_id=st.session_state.chat_id,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            query=prompt,
            response=msg,
            log_dir=os.path.join("src", "front","chat_log")
        )

    except Exception as e:
        error_msg = f"エラーが発生しました: {str(e)}"
        st.session_state.messages.append({"role": "assistant", "content": error_msg})
        st.chat_message("assistant").write(error_msg)
        st.error(f"API呼び出しでエラーが発生しました: {str(e)}")