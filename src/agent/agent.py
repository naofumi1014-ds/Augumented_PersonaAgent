import argparse
import json
import os
import sys
from openai import AzureOpenAI
from dotenv import load_dotenv
from prompts import INITIAL_PERSONA_PROMPT, TOOL_FOR_RETRIEVE_EPISODIC_MEMORY
from request import UserQuery, agentrequest
from tools import get_weather, retrieve_today
from personalization_method import Personalization
from datetime import date, datetime

sys.path.append(
    os.path.join(os.path.dirname(__file__), "../..")
)
from langfuse.decorators import observe
from src.agent.custom_logger import get_custom_logger

load_dotenv()
logger = get_custom_logger(__name__)

class PersonaAgent:
    def __init__(self) -> None:
        self.end_point = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION")
        self.gpt41_deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT")
        
        self.gpt41_model = AzureOpenAI(
            azure_endpoint=str(self.end_point),
            api_key=self.api_key,
            api_version=self.api_version,
        )

    def _clean_message_content(self, content):
        """
        メッセージコンテンツから無効なUnicode文字を除去する
        """
        if isinstance(content, str):
            # サロゲートペアなどの無効なUnicode文字を除去
            return content.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
        return content

    def _clean_messages(self, messages):
        """
        メッセージリスト内の全てのコンテンツをクリーンアップする
        """
        cleaned_messages = []
        for message in messages:
            if isinstance(message, dict):
                cleaned_message = message.copy()
                if 'content' in cleaned_message:
                    cleaned_message['content'] = self._clean_message_content(cleaned_message['content'])
                cleaned_messages.append(cleaned_message)
            else:
                # ChatCompletionMessageオブジェクトの場合は辞書に変換してクリーン
                cleaned_message = {
                    'role': message.role,
                    'content': self._clean_message_content(message.content) if hasattr(message, 'content') else None
                }
                if hasattr(message, 'tool_calls') and message.tool_calls:
                    cleaned_message['tool_calls'] = message.tool_calls
                cleaned_messages.append(cleaned_message)
        return cleaned_messages

    @observe
    def _prompt_personalization(self, system_prompt: str = INITIAL_PERSONA_PROMPT) -> str:
        """
        プロンプトのパーソナライズを実行する
        """
        logger.info("システムプロンプトのパーソナライズを実行します。")
        semantic_memory = self.personalization.generate_semantic_memory()
        logger.info(f"Generated semantic memory: {semantic_memory}")
        return system_prompt.format(Initial_Semantic_Memory=semantic_memory.memories)

    @observe
    def run(self, request: agentrequest) -> str:
        """
        ユーザーに合わせたメモリを取得してエージェントを実行する
        通常のチャットから構築したメモリを基に、パーソナライズしたエージェントを実行する

        Static Workflowで構築 エージェント自体を高度化するのではなくパーソナライズのの有効性を見たい
        """       
        # Personalizationのインスタンスを作成 
        self.personalization = Personalization(request.user)

        # (初回のみ) 比較実験用に、メモリを使うかどうかをトグルする
        if request.use_user_preference_alignment and request.n_of_turns == 1:
            logger.info("初回のシステムプロンプトのパーソナライズを実行します。")
            initial_system_prompt = self._prompt_personalization()
        else:
            initial_system_prompt = None

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "retrieve_today",
                    "description": "今日の日付を取得する",
                    "parameters": {}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "任意の地点の天気予報を取得する",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "都市名（例：Tokyo,Osaka）",
                            },
                            "days": {
                                "type": "integer",
                                "description": "取得する日数（1〜16）",
                                "minimum": 1,
                                "maximum": 16
                            },
                        },
                        "required": ["city","days"],
                    },
                }
            }
        ]

        # 比較実験用に、メモリを使うかどうかをトグルする
        if request.use_user_preference_alignment:
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": "retrieve_user_episodic_memory",
                        "description": TOOL_FOR_RETRIEVE_EPISODIC_MEMORY,
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "The query to retrieve user episodic memory",
                                },
                            },
                            "required": ["query"],
                        },
                    }
                }
            )

        # LLMとの対話およびツール呼び出しのループ処理
        # 比較実験用に、メモリを使うかどうかをトグルする
        if [] == request.messages:
            messages = []
            if request.use_user_preference_alignment:
                request.messages = [{"role":"developer","content":initial_system_prompt},
                        {"role": "user", "content": request.user_query}]
            else: 
                request.messages = [{"role":"developer","content":"一度使ったtoolは使用しないでください。toolの使用が終ったら回答を生成してください。"}
                            ,{"role": "user", "content": request.user_query}]

        
        while True:
            logger.info("LLMにリクエストを送信...")
            # メッセージをクリーンアップしてからリクエスト送信
            cleaned_messages = self._clean_messages(request.messages)

            # 追加のサロゲート除去（念のため二重防御）
            def _strip_invalid_unicode(s):
                if not isinstance(s, str):
                    return s
                return ''.join(ch for ch in s if not (0xD800 <= ord(ch) <= 0xDFFF))
            for m in cleaned_messages:
                if isinstance(m, dict) and 'content' in m and isinstance(m['content'], str):
                    original = m['content']
                    sanitized = _strip_invalid_unicode(original)
                    if original != sanitized:
                        logger.warning("agent.run: サロゲートコードポイントを除去しました。")
                    # encode/decode 念のため
                    m['content'] = sanitized.encode('utf-8','ignore').decode('utf-8','ignore')

            response = self.gpt41_model.chat.completions.create(
                model=self.gpt41_deployment_name,
                messages=cleaned_messages,
                tools=tools,
                tool_choice="auto",
            )
            response_message = response.choices[0].message
            request.messages.append(response_message)

            # logger.info(f"Response from LLM: {response}")
            # 関数呼び出し
            if response_message.tool_calls:
                for tool_call in response_message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    logger.info(f"Function call: {function_name}")  
                    logger.info(f"Function arguments: {function_args}")

                    if function_name == "get_weather":
                        logger.info("天気情報を取得するツールを呼び出しました。")
                        weather_info = []
                        for info in get_weather.GetWeather(city=function_args.get("city"), days=function_args.get("days")).run():
                            d = date.fromisoformat(info["date"]).strftime("%m/%d(%a)")
                            weather_info.append(f"{d}: {info['summary']}  {info['t_min']}°C–{info['t_max']}°C")
                        function_response = "\n".join(weather_info)

                    elif function_name == "retrieve_today":
                        logger.info("現在の時刻を取得するツールを呼び出しました。")
                        function_response = retrieve_today.get_today_date()
                        function_response = f"今日の日付は {function_response} です。"

                    elif function_name == "retrieve_user_episodic_memory":
                        logger.info("Episodic Memoryの取得ツールを呼び出しました。")
                        tool_response = self.personalization.get_episodic_memory(
                            user_query=UserQuery(user_query=function_args.get("query"))
                        )
                        function_response = str(tool_response)

                    request.messages.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": function_response,
                    })
                    # logger.info(f"Function response: {function_response}")
            else:
                logger.info("ツール呼び出しはありませんでした。")

                # 最新のユーザーメッセージを取得
                latest_user_message = ""
                for message in reversed(request.messages):
                    # メッセージが辞書型かオブジェクト型かを判定
                    if isinstance(message, dict):
                        if message["role"] == "user":
                            latest_user_message = message["content"]
                            break
                    else:
                        # ChatCompletionMessageオブジェクトの場合
                        if hasattr(message, 'role') and message.role == "user":
                            latest_user_message = message.content
                            break

                # 会話を自動保存（メッセージが追加されるたびに）
                self.personalization.save_chat_log(
                    chat_id=datetime.now().strftime("%Y-%m-%d"),
                    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    query=latest_user_message,
                    response=response_message.content,
                    log_dir=os.path.join("src", "agent", "memory", "agent")
                )

                request.messages.append({
                    "role": "assistant",
                    "content": response_message.content
                })

                return request.messages

    def optimize(self):
        pass



if __name__ == "__main__":
    agent = PersonaAgent()

    # 1ターン目
    N_OF_TURNS = 1
    USER = "userA"
    USE_USER_PREFERENCE_ALIGNMENT = False

    # 最適化のバッチサイズ
    BATCH_SIZE = 3

    request = agentrequest(
        user=USER,
        user_query="明日から３日間沖縄に行きたいので旅行プランを立ててください。",
        use_user_preference_alignment=USE_USER_PREFERENCE_ALIGNMENT,
        messages=[],
        n_of_turns=N_OF_TURNS
    )
    logger.info(f"Agent request: {request}")
    messages = agent.run(request)
    logger.info(f"Agent response: {messages[-1]}")

    # 2ターン目以降：
    logger.info("2ターン目以降の対話を開始します。")
    while True:

        user_input = input("ユーザーの入力: ")
        if user_input == "exit":
            logger.info("対話を終了します。")
            break

        # D_batchを3として、N_OF_TURNSが3で割り切れる場合は、パーソナライズを行う
        if N_OF_TURNS % BATCH_SIZE == 0:
            personalization_result = Personalization(user=USER).user_preference_alignment(batch_size=BATCH_SIZE)
            # 最初のシステムメッセージと差し替える
            logger.info("システムメッセージの最適化を行いました。")

            # 確認用に保存する iがあれば＋１する
            # 今日の日付
            today = date.today().strftime("%Y%m%d")
            i = 0
            while os.path.exists(os.path.join("src", "PersonaPrompt", f"personalization_result_{today}_{i}.txt")):
                i += 1

            # フォルダは新規作成
            os.makedirs(os.path.join("src", "agent", "PersonaPrompt"), exist_ok=True)
            with open(os.path.join("src", "agent", "PersonaPrompt", f"personalization_result_{today}_{i}.txt"), "w") as f:
                json.dump(personalization_result, f, ensure_ascii=False, indent=4)

            messages[0] = {
                "role": "developer",
                "content": personalization_result
            }

        messages.append({
            "role": "user",
            "content": user_input
        })

        N_OF_TURNS += 1
        request = agentrequest(
            user=USER,
            user_query=user_input, #ログ保存用
            use_user_preference_alignment=USE_USER_PREFERENCE_ALIGNMENT,
            messages=messages,
            n_of_turns=N_OF_TURNS
        )
        # logger.info(f"Agent request: {request}")
        messages = agent.run(request)
        logger.info(f"Agent response: {messages[-1]}")