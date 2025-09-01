import json
import re
import unicodedata
from dotenv import load_dotenv
from src.agent.request import chat_log_by_chat_id, UserQuery
from src.agent.custom_logger import get_custom_logger
from src.agent.response import Episode, Aux, Chat_log, Episode_from_PA,EpisodicMemory, EpisodicMemory_from_PA, SemanticMemory
from src.agent.prompts import INITIAL_PERSONA_PROMPT,COMPUTE_LOSS_GRADIENT_PROMPT,GRADIENT_UPDATE_PROMPT
import os
import faiss
import numpy as np

load_dotenv()
logger = get_custom_logger(__name__)

use_langfuse = os.getenv("USE_LANGFUSE", False)
os.environ["LANGFUSE_PUBLIC_KEY"] = os.getenv("LANGFUSE_PUBLIC_KEY", "")
os.environ["LANGFUSE_SECRET_KEY"] = os.getenv("LANGFUSE_SECRET_KEY", "")
if use_langfuse:
    logger.info("langfuse is used")
    from langfuse.openai import AzureOpenAI
else:
    logger.info("langfuse is not used")
    from openai import AzureOpenAI

from langfuse.decorators import observe

class Personalization:
    """
    チャットのログからsemantic memoryとepisodic memoryを生成するクラス
    """

    def __init__(self, user: str) -> None:
        self.user = user

        self.end_point = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION")
        self.gpt41_deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT")

        self.embedding_api_version = os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION")
        self.embedding_deployment_name = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

        self.ai_search_key = os.getenv("AI_SEARCH_KEY")
        self.ai_search_endpoint = os.getenv("AI_SEARCH_ENDPOINT")

        self.o4_mini_endpoint = os.getenv("AZURE_OPENAI_o4_mini_ENDPOINT")
        self.o4_mini_deployment_name = os.getenv("AZURE_OPENAI_o4_mini_DEPLOYMENT")
        self.o4_mini_api_key = os.getenv("AZURE_OPENAI_o4_mini_API_KEY")
        self.o4_mini_api_version = os.getenv("AZURE_OPENAI_o4_mini_API_VERSION")

        self.gpt41_model = AzureOpenAI(
            azure_endpoint=str(self.end_point),
            api_key=self.api_key,
            api_version=self.api_version,
            azure_deployment=self.gpt41_deployment_name,
        )

        self.o4_mini_model = AzureOpenAI(
            azure_endpoint=str(self.o4_mini_endpoint),
            api_key=self.o4_mini_api_key,
            api_version=self.o4_mini_api_version,
            azure_deployment=self.o4_mini_deployment_name,
        )

        self.embedding = AzureOpenAI(
            azure_endpoint=str(self.end_point),
            api_key=self.api_key,
            api_version=self.embedding_api_version,
            azure_deployment=self.embedding_deployment_name,
        )

    def _load_episodic_memory(self, user: str) -> EpisodicMemory_from_PA:
        """
        ユーザー名からエピソードメモリを全て読み込む
        """
        memory_dir = os.path.join("src", "agent","memory", "episodic")
        semantic_memory_file = os.path.join(memory_dir, f"episodic_memory_{user}.json")

        if not os.path.exists(semantic_memory_file):
            logger.warning(f"Episodic memory file does not exist: {semantic_memory_file}")
            return EpisodicMemory_from_PA(episodes=[])

        with open(semantic_memory_file, 'r', encoding='utf-8') as f:
            content = f.read()
            if not content:
                logger.warning(f"Episodic memory file is empty: {semantic_memory_file}")
                return EpisodicMemory_from_PA(episodes=[])

            episodic_memory_data = json.loads(content)
            episodes = [Episode_from_PA(**episode) for episode in episodic_memory_data['episodes']]
            return EpisodicMemory_from_PA(episodes=episodes)
        
    def _load_agent_interactions(self, user: str) -> EpisodicMemory_from_PA:
        """
        ユーザー名からエージェントとのインタラクションを全て読み込む
        """
        memory_dir = os.path.join("src", "agent","memory", "agent")
        agent_interaction_file = os.path.join(memory_dir, f"chat_log_{user}.json")

        if not os.path.exists(agent_interaction_file):
            logger.warning(f"Agent interaction file does not exist: {agent_interaction_file}")
            return EpisodicMemory_from_PA(episodes=[])

        with open(agent_interaction_file, 'r', encoding='utf-8') as f:
            content = f.read()
            if not content:
                logger.warning(f"Agent interaction file is empty: {agent_interaction_file}")
                return EpisodicMemory_from_PA(episodes=[])

            memory_data = json.loads(content)
            # memory_dataは配列形式なので、直接それをエピソードとして扱う
            episodes = []
            for interaction in memory_data:
                # Chat_logからEpisode_from_PAに変換
                episode = Episode_from_PA(
                    user=user,
                    query=interaction.get('query', ''),
                    response=interaction.get('response', ''),
                    rili=interaction.get('rili', ''),
                    embeddings=interaction.get('embeddings', []),
                    aux=Aux(timestamp="", chat_id="")  # auxは空のAuxで初期化
                )
                episodes.append(episode)
            return EpisodicMemory_from_PA(episodes=episodes)

    @observe
    def generate_rili_single(self,query,response) -> str:
        # --- Unicodeサニタイズ ---
        def _strip_invalid_unicode(s):
            if not isinstance(s, str):
                return s
            # UTF-16サロゲート領域の除去
            cleaned = ''.join(ch for ch in s if not (0xD800 <= ord(ch) <= 0xDFFF))
            if cleaned != s:
                logger.warning("generate_rili_single: サロゲートコードポイントを除去しました。")
            # 念のため再エンコード/デコードで不正シーケンス除去
            try:
                cleaned = cleaned.encode('utf-8', 'ignore').decode('utf-8', 'ignore')
            except Exception as e:
                logger.error(f"Unicodeサニタイズ中に例外: {e}")
            return cleaned

        query = _strip_invalid_unicode(str(query))
        response = _strip_invalid_unicode(str(response))

        INFER_TRUE_RESPONSE_PROMPT = """
        あなたはユーザーとLLMの会話ログを管理する監督者です。
        会話ログをもとに、ユーザーが求めていたであろう真の応答を生成してください。
        ユーザーのクエリに対して、LLMがどのような応答を返すことで、ユーザーが本当に聞きたかったことについて回答できるか、を考慮してください。

        # 真の応答を生成する上で、推論する観点
        - 会話の前後から、ユーザーが本当に知りたかったことは何か
        - LLMの応答がユーザーの期待に応えているかどうか
        - ユーザーのクエリに対して、矛盾なく応答できているか

        LLMの応答が適切であった場合は会話の内容をそのまま返してください。
        ただし、LLMの応答が不適切であったと考えられる場合は、本来ユーザーが求めていたであろう情報が含まれた真の応答を生成してください。

        回答を推論する際は、会話の前後の文脈やユーザーの反応を考慮してください。
        """

        USER_PROMPT = f"""
        以下の会話ログをもとに、ユーザーが求めていたであろう真の応答を生成してください。
        
        # 応答生成の対象となる会話ログ
        - ユーザークエリ：{query}
        - LLMの応答：{response}
        """

        messages = [
                {"role": "system", "content": INFER_TRUE_RESPONSE_PROMPT},
                {
                    "role": "user",
                    "content": USER_PROMPT,
                },
        ]

        response = self.gpt41_model.chat.completions.create(
            model=self.gpt41_deployment_name,
            messages=messages,
        )

        return response.choices[0].message.content

    @observe
    def generate_rili(self, chat_log_by_chat_id: chat_log_by_chat_id) -> EpisodicMemory:
        """
        ri_gt(ground truth)は実応用において取得不可能なため
        ri_li(llm inference)を近似的に生成する
        """

        INFER_TRUE_RESPONSE_PROMPT = """
        あなたはユーザーとLLMの会話ログを管理する監督者です。
        会話ログをもとに、ユーザーが求めていたであろう真の応答を生成してください。
        ユーザーのクエリに対して、LLMがどのような応答を返すことで、ユーザーが本当に聞きたかったことについて回答できるか、を考慮してください。

        # 真の応答を生成する上で、推論する観点
        - 会話の前後から、ユーザーが本当に知りたかったことは何か
        - LLMの応答がユーザーの期待に応えているかどうか
        - ユーザーのクエリに対して、矛盾なく応答できているか
        - 会話が複数ターンに及んでいないか（複数の場合、文脈によってはほしい情報が得られておらず何度も聞き返している）

        LLMの応答が適切であった場合は会話の内容をそのまま返してください。
        ただし、LLMの応答が不適切であったと考えられる場合は、本来ユーザーが求めていたであろう情報が含まれた真の応答を生成してください。

        回答を推論する際は、会話の前後の文脈やユーザーの反応を考慮してください。
        """

        logger.info("riliの推定を開始")

        # Unicodeサニタイズ用関数（singleと共有）
        def _strip_invalid_unicode(s):
            if not isinstance(s, str):
                return s
            cleaned = ''.join(ch for ch in s if not (0xD800 <= ord(ch) <= 0xDFFF))
            if cleaned != s:
                logger.warning("generate_rili: サロゲートコードポイントを除去しました。")
            return cleaned.encode('utf-8', 'ignore').decode('utf-8', 'ignore')

        episodic_memory = EpisodicMemory(episodes=[])
        for each in chat_log_by_chat_id:
            # eachが辞書想定
            safe_each = {}
            if isinstance(each, dict):
                for k,v in each.items():
                    if isinstance(v, str):
                        safe_each[k] = _strip_invalid_unicode(v)
                    else:
                        safe_each[k] = v
            else:
                safe_each = each
            USER_PROMPT = f"""
            以下の会話ログをもとに、ユーザーが求めていたであろう真の応答を生成してください。
            
            # 会話全体のログ
            {chat_log_by_chat_id}

            # 応答生成の対象となる会話ログ
            {safe_each}
            """
            
            messages = [
                {"role": "system", "content": INFER_TRUE_RESPONSE_PROMPT},
                {
                    "role": "user",
                    "content": USER_PROMPT,
                },
            ]

            response = self.gpt41_model.beta.chat.completions.parse(
                model=self.gpt41_deployment_name,
                messages=messages,
                response_format=Episode,
            )

            # エピソードメモリにriliを追加（parsed_contentを使用してEpisodeオブジェクトを取得）
            parsed_episode: Episode = response.choices[0].message.parsed
            episodic_memory.episodes.append(parsed_episode)
            
        logger.info("riliの推定を終了")

        return episodic_memory
    
    @observe
    def generate_embeddings(self, episodic_memory:EpisodicMemory) -> EpisodicMemory:
        """
        エピソードメモリから[質問]-[応答]の埋め込みベクトルを生成するメソッド
        """

        logger.info("埋め込みベクトルの生成を開始")

        for episode in episodic_memory.episodes:
            # [質問]-[応答]の埋め込みベクトルを生成
            embedding = self.embedding.embeddings.create(
                model=self.embedding_deployment_name,
                input=[f"[ユーザークエリ]{episode.query} - [LLM応答]{episode.rili}"]
            )
            episode.embeddings = embedding.data[0].embedding

        logger.info("埋め込みベクトルの生成を終了")

        return episodic_memory


    def generate_episodic_memory(self, chat_id: str, user: str) -> None:
        """
        チャットのログからepisodic memoryを生成するメソッド
        """
        logger.info(f"エピソードメモリの生成を開始 chat_id: {chat_id}, user: {user}")

        # チャットログパス
        log_dir = os.path.join("src", "front","chat_log")
        log_file = os.path.join(log_dir, f"chat_log_{self.user}.json")

        # セマンティックメモリパス
        memory_dir = os.path.join("src", "agent","memory", "episodic")
        os.makedirs(memory_dir, exist_ok=True)
        semantic_memory_file = os.path.join(memory_dir, f"episodic_memory_{self.user}.json")

        #ログファイルを確認し、エピソードメモリを生成する
        if os.path.exists(log_file):
            with open(log_file, 'r', encoding='utf-8') as f:
                chat_log = json.load(f)
                chat_log_by_chat_id = []

                for episode in chat_log:
                    #chat_idが一致するエピソードを抽出
                    if episode['aux']['chat_id'] == chat_id:
                        chat_log_by_chat_id.append(episode)
                
                #! ri_liの生成
                episodic_memory = self.generate_rili(chat_log_by_chat_id)

        #! メモリ取得時に使用する埋め込みベクトルの生成
        episodic_memory = self.generate_embeddings(episodic_memory)

        # 既存のデータを読み込み
        existing_data = []
        if os.path.exists(semantic_memory_file):
            try:
                with open(semantic_memory_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        existing_data = json.loads(content)
            except json.JSONDecodeError:
                # JSONが破損している場合は空のリストから開始
                existing_data = []

        # 新しいエピソードメモリをEpisodicMemory型として保存
        episodic_data = episodic_memory.model_dump()
        
        # 既存データと新しいデータをマージ（EpisodicMemory形式を保持）
        if existing_data and isinstance(existing_data, dict) and 'episodes' in existing_data:
            # 既存のエピソードを処理（文字列形式の場合はパース）
            existing_episodes = []
            for episode in existing_data['episodes']:
                if isinstance(episode, str):
                    # 文字列形式の場合はJSONとしてパース
                    try:
                        parsed_episode = json.loads(episode)
                        existing_episodes.append(parsed_episode)
                        logger.info(f"文字列形式のエピソードをパースしました: {parsed_episode}")
                    except json.JSONDecodeError:
                        logger.warning(f"パースできない文字列エピソード: {episode}")
                elif isinstance(episode, dict):
                    # 既にオブジェクト形式の場合はそのまま使用
                    existing_episodes.append(episode)
                else:
                    logger.warning(f"予期しないエピソード形式: {type(episode)}, {episode}")
            
            # 既存のエピソードと新しいエピソードをマージ
            episodic_data['episodes'].extend(existing_episodes)
        
        # EpisodicMemory型でJSONファイルとして保存
        # 保存前に念のためサニタイズ（絵文字 & サロゲート除去）
        def _remove_surrogates(s: str) -> str:
            return ''.join(ch for ch in s if not (0xD800 <= ord(ch) <= 0xDFFF))
        emoji_pattern = re.compile('[\U0001F300-\U0001FAFF\U00002700-\U000027BF\U0001F1E6-\U0001F1FF\U00002600-\U000026FF\U0001F900-\U0001F9FF]+')
        def _sanitize_text(s: str) -> str:
            if not isinstance(s, str):
                return s
            s = _remove_surrogates(s)
            s = emoji_pattern.sub('', s)
            return s.encode('utf-8','ignore').decode('utf-8','ignore')
        def _sanitize(obj):
            if isinstance(obj, dict):
                return {k: _sanitize(v) for k,v in obj.items()}
            if isinstance(obj, list):
                return [_sanitize(v) for v in obj]
            if isinstance(obj, str):
                return _sanitize_text(obj)
            return obj
        episodic_data = _sanitize(episodic_data)
        with open(semantic_memory_file, 'w', encoding='utf-8') as f:
            json.dump(episodic_data, f, ensure_ascii=False, indent=2)

        logger.info(f"エピソードメモリの生成を終了 chat_id: {chat_id}, user: {user}")

    def save_chat_log(self,chat_id,timestamp,query,response,log_dir) -> None:
        """
        チャットのログを生成する
        """

        # --- 文字列サニタイズ用ヘルパ ---
        def _remove_surrogates(s: str) -> str:
            # UTF-16サロゲート範囲を除去
            return ''.join(ch for ch in s if not (0xD800 <= ord(ch) <= 0xDFFF))

        # よく使う絵文字のブロックを正規表現で除去
        emoji_pattern = re.compile('[\U0001F300-\U0001FAFF\U00002700-\U000027BF\U0001F1E6-\U0001F1FF\U00002600-\U000026FF\U0001F900-\U0001F9FF]+')

        def _remove_emojis(s: str) -> str:
            return emoji_pattern.sub('', s)

        def _sanitize_text(s: str) -> str:
            if not isinstance(s, str):
                return s
            s = _remove_surrogates(s)
            s = _remove_emojis(s)
            # 念のためエンコード失敗文字を除去
            s = s.encode('utf-8', 'ignore').decode('utf-8', 'ignore')
            return s

        def _sanitize(obj):
            if isinstance(obj, dict):
                return {k: _sanitize(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_sanitize(v) for v in obj]
            if isinstance(obj, str):
                return _sanitize_text(obj)
            return obj

        # ディレクトリの作成
        os.makedirs(log_dir, exist_ok=True)

        # 会話ログファイルのパス
        log_file = os.path.join(log_dir, f"chat_log_{self.user}.json")

        # auxの構築
        aux: Aux = Aux(
            timestamp=timestamp,
            chat_id=chat_id
        )

        # riliの生成
        rili = self.generate_rili_single(query,response)

        # エピソードメモリ(response 未成形)の構築
        conversation_data: Episode_from_PA = Episode_from_PA(
            user=self.user,
            query=query,
            response=response,
            aux=aux,
            rili=rili # riliの結果を注入する
        )

        # 既存のデータを読み込み
        episodes = []
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        episodes = json.loads(content)
            except json.JSONDecodeError:
                # JSONが破損している場合は空のリストから開始
                episodes = []

        # 新しいエピソードを追加（サニタイズ）
        sanitized_episode = _sanitize(conversation_data.model_dump())
        episodes.append(sanitized_episode)

        # JSONファイルとして保存（配列形式）
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(_sanitize(episodes), f, ensure_ascii=False, indent=2)

    def get_episodic_memory(self, user_query: UserQuery, top_k: int=5) -> EpisodicMemory_from_PA:
        """
        Top-Kのエピソードメモリを取得する
        エージェントのtoolとして使用するためのメソッド
        embeddingベクトルを使用して、ユーザーのクエリに関連するエピソードメモリを取得する

        #  Args:
            user_query (UserQuery): ユーザーのクエリ
            top_k (int): 取得するエピソードメモリの数
        #  Returns:
            EpisodicMemory: エピソードメモリのリスト
        """
        logger.info(f"ユーザー: {self.user} のエピソードメモリを取得開始")
        # ユーザークエリをベクトル化
        embedding_user_query = self.embedding.embeddings.create(
            model=self.embedding_deployment_name,
            input=[user_query.user_query]
        )

        # ユーザークエリの埋め込みベクトル
        user_query_embedding = embedding_user_query.data[0].embedding

        # エピソーディックメモリに対してtop_kの類似度検索
        memory_dir = os.path.join("src", "agent","memory", "episodic")
        semantic_memory_file = os.path.join(memory_dir, f"episodic_memory_{self.user}.json")
        if not os.path.exists(semantic_memory_file):
            logger.warning(f"Episodic memory file does not exist: {semantic_memory_file}")
            return ValueError(f"Episodic memory file does not exist: {semantic_memory_file}")
        
        with open(semantic_memory_file, 'r', encoding='utf-8') as f:
            content = f.read()
            if not content:
                logger.warning(f"Episodic memory file is empty: {semantic_memory_file}")
                return EpisodicMemory_from_PA(episodes=[])

            episodic_memory_data = json.loads(content)
            embeddings = np.array([each['embeddings'] for each in episodic_memory_data['episodes']],dtype="float32")

        # FAISS インデックス作成
        d = embeddings.shape[1]       # 次元数
        index = faiss.IndexFlatIP(d)  # コサイン類似度（内積）用
        faiss.normalize_L2(embeddings)
        index.add(embeddings)

        # ユーザークエリの埋め込みベクトルを正規化
        faiss.normalize_L2(np.array([user_query_embedding], dtype="float32"))
        scores, ids = index.search(np.array([user_query_embedding], dtype="float32"), k=top_k)
        
        logger.info(f"Top {top_k} episodes retrieved")
        # top_kのエピソードメモリを取得
        top_k_episodes = EpisodicMemory_from_PA(episodes=[])
        for i in ids[0]:
            episode = episodic_memory_data['episodes'][i]
            logger.info(f"Retrieved episode: {episode['query']}")
            # Episodeオブジェクトに変換
            episode = Episode_from_PA(**episode)
            top_k_episodes.episodes.append(episode)

        logger.info(f"Retrieved {len(top_k_episodes.episodes)} episodes for user: {self.user}")
        return top_k_episodes
    
    @observe
    def generate_semantic_memory(self) -> SemanticMemory:
        """
        エージェントの行動に沿うように、ユーザー個別のセマンティックメモリを生成する
        """
        logger.info("Generating Semantic Memory...")

        PROMPT = """
        # Role
        あなたは旅行というトピックに対し、ユーザーの好みを推測するためのアシスタントです。

        # Instructions
        会話ログからユーザーの個性を抽出し、セマンティックメモリを生成してください。

        # Information
        - ユーザーとLLMの過去の会話：{episodic_memory}

        # Return Format
        - SemanticMemory = [
        "- 性別や年齢層、どんな仕事をしていそうか（例：30代の男性で、IT企業に勤めている）",
        "- 好きなジャンルやテーマ（例：歴史映画やドキュメンタリーを好む）",
        "- コミュニケーションスタイル（例：簡潔で直接的な回答を好む）",
        "- 思考の焦点（例：ダークコメディや社会風刺的作品への関心が高い）",
        "- 回答の癖（例：質問に対して詳細な説明を求める傾向がある）",
        "- 言葉遣い（例：カジュアルな言葉遣いを好む）",
        "- ユーザーの質問に対する応答の傾向（例：質問に対して具体的な例を求めることが多い）",
        etc...(あなた自身が考え、ユーザーの個性をまとめてください)
        ]
        """

        # 全てのエピソーディックメモリを取得

        episodic_memory_from_pa = self._load_episodic_memory(self.user)
        if not episodic_memory_from_pa.episodes:
            logger.warning(f"No episodic memory found for user: {self.user}")
        episodic_memory_from_pa_str = json.dumps(episodic_memory_from_pa.model_dump(), ensure_ascii=False, indent=2)

        response = self.gpt41_model.beta.chat.completions.parse(
                model=self.gpt41_deployment_name,
                messages=[
                    {"role": "user", "content": PROMPT.format(episodic_memory=episodic_memory_from_pa_str)}],
                response_format=SemanticMemory
        )

        return response.choices[0].message.parsed

    @observe
    def optimization(self,D_batch: EpisodicMemory_from_PA ,persona_prompt: str, user: str) -> str:
        """
        論文提案手法のペルソナの最適化を行う

        #  Args:
            Test User Data D : EpisodicMemory
            Initial persona P : str
        #  Returns:
            Optimized persona P*
        """
        # ! procedure Optimization(D_batch, P)
        logger.info("プロンプトの更新を開始")
        grads = [] # ∇^^
        for episode in D_batch.episodes: #batchサイズ分取得
            
            #Compute ∇ ← LLM_grad(episode)
            response = self.o4_mini_model.chat.completions.create(
                model=self.o4_mini_deployment_name,
                messages=[
                    {"role": "user", "content": COMPUTE_LOSS_GRADIENT_PROMPT.format(question=episode.query, expected_answer=episode.rili, agent_response=episode.response)}
                ]
            )

            grad = response.choices[0].message.content
            grads.append(grad)

        # Gradient Update P* ← LLM_update(P, ∇^^)
        response = self.o4_mini_model.chat.completions.create(
            model=self.o4_mini_deployment_name,
            messages=[
                {"role": "user", "content": GRADIENT_UPDATE_PROMPT.format(current_system_prompt=persona_prompt, aggregated_feedback=str(grads))}
            ]
        )

        self.updated_persona_prompt = response.choices[0].message.content
        logger.info("プロンプトの更新を終了")

        return self.updated_persona_prompt


    def user_preference_alignment(self,batch_size:int) -> str:
        """
        ユーザーの好みをペルソナに反映させるためのメソッド
        """
        logger.info("プロンプトの個人最適化を開始")

        EPSILON = 3  # ε
        logger.info("params: ε = {EPSILON}, BATCH_SIZE = {BATCH_SIZE}".format(EPSILON=EPSILON, BATCH_SIZE=batch_size))

        # ユーザー名からjsonを取得
        # ! 呼び出すたびに、履歴のバッチの最新Ｎ件を取得する
        D: EpisodicMemory_from_PA = self._load_agent_interactions(self.user)
        logger.info(D.episodes[:batch_size])
        persona_prompt = INITIAL_PERSONA_PROMPT # 初期ペルソナプロンプト iterの1回目のみ
        for i in range(EPSILON):
            logger.info(f"iteration : {i+1}/{EPSILON}")

            D_batch = EpisodicMemory_from_PA(episodes=D.episodes[:batch_size])  # バッチサイズ分のエピソードを取得  
            persona_prompt = self.optimization(D_batch, persona_prompt, self.user)

        self.optimized_persona_prompt = persona_prompt

        logger.info("プロンプトの個人最適化を終了")
        return self.optimized_persona_prompt
