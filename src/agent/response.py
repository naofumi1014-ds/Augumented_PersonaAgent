from typing import Any
from pydantic import BaseModel, Field
from typing import Optional


'''
Episodic Memory Response Model
'''
class Aux(BaseModel):
    timestamp: str = Field(..., description="メッセージのタイムスタンプ")
    chat_id: str = Field(..., description="チャットID")

class Chat_log(BaseModel):
    user: str = Field(..., description="ユーザー情報")
    query: str = Field(..., description="クエリ")
    response: str = Field(..., description="LLMの応答")
    aux: Aux = Field(..., description="メタ情報")

class Episode(BaseModel):
    user: str = Field(..., description="ユーザー情報")
    query: str = Field(..., description="クエリ")
    rili: str = Field(..., description="LLMの推論によって生成された真の応答")

    # response may be missing in some records; default to empty string
    response: str = Field("", description="LLMの応答")
    aux: Aux = Field(..., description="メタ情報")
    embeddings: list[float] = Field(..., description="[質問]-[応答]の埋め込みベクトル")

class EpisodicMemory(BaseModel):
    episodes: list[Episode] = Field(...,description="エピソードのリスト")

class Episode_from_PA(BaseModel):
    user: str = Field(..., description="ユーザー情報")
    query: str = Field(..., description="クエリ")
    rili: str = Field(..., description="LLMの推論によって生成された真の応答")
    response: str = Field(..., description="LLMの応答")
    aux: Optional[Aux] = Field(..., description="メタ情報")

class EpisodicMemory_from_PA(BaseModel):
    episodes: list[Episode_from_PA] = Field(..., description="エピソードのリスト")

class SemanticMemory(BaseModel):
    memories: list[str] = Field(..., description="セマンティックメモリのリスト")

class agentresponse(BaseModel):
    messages :list[dict[str, Any]]  = Field(..., description="LLMとユーザーのメッセージ")