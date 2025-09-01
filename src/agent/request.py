from typing import Literal
from pydantic import BaseModel, Field
import os
import sys
sys.path.append(
    os.path.join(os.path.dirname(__file__), "../..")
)

from src.agent.response import Chat_log


class chat_log_by_chat_id(BaseModel):
    chat_log: list[Chat_log]= Field(..., description="チャットIDが共通のチャットログ")


class UserQuery(BaseModel):
    user_query: str = Field(..., description="ユーザーのクエリ")


class agentrequest(BaseModel):
    user:str = Field(..., description="ユーザー名")
    user_query:str = Field(..., description="ユーザーのクエリ")
    use_user_preference_alignment: bool = Field(..., description="ユーザーの好みを考慮するかどうか")
    messages: list = Field(default=[], description="メッセージのリスト")
    n_of_turns: int = Field(default=0, description="会話のターン数")