"""
RAGに関連するクラスと関数を定義
"""

import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from langchain.schema.runnable import Runnable
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores.base import VectorStore
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_fixed

from fdua_competition.enums import ChatModelOption
from fdua_competition.utils import print_before_retry
from fdua_competition.vectorstore import retrieve_context


def get_chat_model(opt: ChatModelOption) -> ChatOpenAI:
    """
    指定されたchatモデルオプションに基づいてchatモデルを取得する
    args:
        opt (ChatModelOption): 使用するchatモデルのオプション
    returns:
        ChatOpenAI: 選択されたchatモデルのインスタンス
    raises:
        ValueError: 未知のモデルオプションが指定された場合
    """
    match opt:
        case ChatModelOption.AZURE:
            return AzureChatOpenAI(azure_deployment="4omini")
        case _:
            raise ValueError(f"unknown model: {opt}")


def read_prompt(target: str) -> str:
    """
    プロンプトファイルからテキストを読み込む
    args:
        target (str): プロンプトファイル名（拡張子なし）
    returns:
        str: プロンプトの内容
    """
    return (Path(__file__).parent / f"prompts/{target}.txt").read_text()


# [start: base rag class]
@dataclass
class RAG(ABC):
    """
    RAGの基底クラス
    attributes:
        vectorstore (VectorStore): 使用するvectorstore
        chat_model_option (ChatModelOption): 使用するchatモデルのオプション(デフォルトはAZURE)
    """

    vectorstore: VectorStore
    chat_model_option: ChatModelOption = ChatModelOption.AZURE

    @property
    @abstractmethod
    def prompt_template(self) -> ChatPromptTemplate:
        """
        プロンプトテンプレートを返すプロパティ. サブクラスで実装する.
        * プレースホルダを含める -> {query}, {context}
        """
        ...

    @abstractmethod
    def build_payload(self, query: str) -> dict[str, t.Any]:
        """
        クエリに応じたペイロードを構築するプロパティ. サブクラスで実装する
        args:
            query (str): ユーザーからの質問
        returns:
            dict[str, Any]: ペイロードの辞書
        """
        ...

    @property
    @abstractmethod
    def output_structure(self) -> BaseModel:
        """
        出力構造 (Pydanticモデル)を返すプロパティ. サブクラスで実装する
        """
        ...

    @property
    def chat_model(self) -> ChatOpenAI:
        """
        構造化出力を設定したchatモデルを返すプロパティ
        returns:
            ChatOpenAI: 構造化出力を持つchatモデルのインスタンス
        """
        return get_chat_model(self.chat_model_option).with_structured_output(self.output_structure)

    @property
    def chain(self) -> Runnable:
        """
        プロンプトテンプレートとchatモデルを連結したチェーンを返すプロパティ
        returns:
            Runnable: チェーンのインスタンス
        """
        return self.prompt_template | self.chat_model

    @retry(stop=stop_after_attempt(24), wait=wait_fixed(1), before_sleep=print_before_retry)
    def invoke(self, query: str) -> BaseModel:
        """
        クエリに対してRAGチェーンを実行し回答を取得する
        args:
            query (str): ユーザーからの質問
        returns:
            BaseModel: 構造化された回答を含むPydanticモデル
        """
        return self.chain.invoke({
            **self.build_payload(query), 
            "context": retrieve_context(vectorstore=self.vectorstore, query=query)
        })


# [end: base rag class]


# [start: research_assistant]
class ResearchAssistantResponse(BaseModel):
    query: str = Field(description="the query that was asked.")
    response: str = Field(description="the answer for the given query")
    reason: str = Field(description="the reason for the response.")
    organization_name: str = Field(description="the organization name that the query is about.")
    contexts: list[str] = Field(description="the context that the response was based on with its file path and page number.")


class ResearchAssistant(RAG):
    def __init__(
        self, vectorstore: VectorStore, chat_model_option: ChatModelOption = ChatModelOption.AZURE, language: str = "japanese"
    ):
        super().__init__(vectorstore=vectorstore, chat_model_option=chat_model_option)
        self.language = language

    @property
    def prompt_template(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages(
            [
                ("system", "{system_prompt}"),
                ("system", "context:\n{context}"),
                ("user", "query:\n{query}"),
            ]
        )

    def build_payload(self, query: str) -> dict[str, t.Any]:
        return {
            "system_prompt": read_prompt("research_assistant"),
            "query": query, 
            "language": self.language
        }

    @property
    def output_structure(self) -> BaseModel:
        return ResearchAssistantResponse


# [end: research_assistant]
