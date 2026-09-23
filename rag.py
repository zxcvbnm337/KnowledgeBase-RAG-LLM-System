from __future__ import annotations

from typing import Iterator, Optional

from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory

import config_data as config
from file_history_store import FileChatMessageHistory
from retrieval import DashScopeReranker, RetrievalOutcome, RetrievalPipeline
from vector_stores import VectorStoreService


def print_prompt(prompt):
    if not config.debug_print_prompt:
        return prompt
    print("=" * 20)
    print(prompt.to_string())
    print("=" * 20)
    return prompt


class RagService(object):
    def __init__(self):

        self.vector_service = VectorStoreService(
            embedding=DashScopeEmbeddings(model=config.embedding_model_name)
        )

        # 检索链路：向量召回 → 交叉编码重排 → 阈值门控（见 retrieval.py）
        # 关闭重排时传 None，管线会自动改用向量相似度阈值判定。
        self.pipeline = RetrievalPipeline(
            search_fn=self.vector_service.search_with_scores,
            reranker=DashScopeReranker(model=config.rerank_model) if config.enable_rerank else None,
            top_k=config.retrieval_top_k,
            final_k=config.retrieval_final_k,
            rerank_threshold=config.rerank_threshold,
            vector_similarity_threshold=config.vector_similarity_threshold,
            refusal_text=config.refusal_text,
        )

        # 最近一次检索的中间状态，供页面展示（可观测性）
        self.last_retrieval: Optional[RetrievalOutcome] = None

        # 检索不命中时的统一答复文案
        self.refusal_text = config.refusal_text

        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", "以我提供的已知参考资料为主，"
                 "简洁和专业的回答用户问题。参考资料:{context}。"),
                ("system", "并且我提供用户的对话历史记录，如下："),
                MessagesPlaceholder("history"),
                ("user", "请回答用户提问：{input}")
            ]
        )

        self.chat_model = ChatTongyi(model=config.chat_model_name)

        # 专门用于压缩历史摘要的模型：不参与对话，只做"旧对话 -> 摘要"这一步
        self.summary_model = ChatTongyi(model=config.memory_summary_model)
        self.summary_prompt = ChatPromptTemplate.from_template(config.memory_summary_prompt)

        self.chain = self.__get_chain()

    # ==================== 对话记忆（分层） ====================
    def _summarize_history(self, previous_summary: str, dialogue: str) -> str:
        """把「已有摘要 + 一批更早的对话」合并压缩成新的摘要。"""
        summary_chain = self.summary_prompt | self.summary_model | StrOutputParser()
        return summary_chain.invoke({
            "previous_summary": previous_summary.strip() or "（暂无）",
            "dialogue": dialogue,
        })

    def _get_history(self, session_id: str) -> FileChatMessageHistory:
        """框架工厂：每次调用返回一个带摘要能力的会话历史对象。

        RunnableWithMessageHistory 会在每次 invoke 前后自动用它读写历史，
        并把 messages 属性（摘要 + 最近若干条原文）注入 prompt 的 history 占位符。
        """
        return FileChatMessageHistory(
            session_id,
            storage_path=config.chat_history_dir,
            recent_messages=config.memory_recent_messages,
            summary_trigger=config.memory_summary_trigger,
            summarizer=self._summarize_history,
        )

    def memory_stats(self, session_id: str) -> dict:
        """读取某个会话的分层记忆状态，供页面展示。"""
        return FileChatMessageHistory(
            session_id, storage_path=config.chat_history_dir
        ).stats()

    # ==================== 检索问答链 ====================
    def __get_chain(self):
        """获取最终的执行链。

        参考资料（context）由 `answer()` / `stream()` 在链外检索后注入，
        而不是在链内用 retriever 拉取——这样「检索不命中就直接拒答」
        才能在调用大模型之前短路。
        """
        chain = (
            self.prompt_template
            | print_prompt
            | self.chat_model
            | StrOutputParser()
        )

        conversation_chain = RunnableWithMessageHistory(       # 增强的链
            chain,
            self._get_history,
            input_messages_key="input",
            history_messages_key="history",
        )

        return conversation_chain

    # ==================== 对外问答接口（带门控） ====================
    def retrieve(self, question: str) -> RetrievalOutcome:
        """执行一次完整检索，并把中间状态留档供页面展示。"""
        outcome = self.pipeline.retrieve(question)
        self.last_retrieval = outcome
        return outcome

    def _should_refuse(self, outcome: RetrievalOutcome) -> bool:
        return bool(outcome.refused and config.refusal_short_circuit)

    def answer(self, question: str, session_config: dict = None) -> str:
        """非流式问答。检索不命中时直接返回拒答文案，不调用大模型。"""
        outcome = self.retrieve(question)
        if self._should_refuse(outcome):
            # 刻意不写入对话历史：拒答既不含知识、也会污染后续的摘要压缩
            return self.refusal_text
        return self.chain.invoke(
            {"input": question, "context": outcome.context_text()},
            session_config or config.session_config,
        )

    def stream(self, question: str, session_config: dict = None) -> Iterator[str]:
        """流式问答。拒答场景下同样以流的形式吐出文案，前端无需特殊分支。"""
        outcome = self.retrieve(question)
        if self._should_refuse(outcome):
            yield self.refusal_text
            return
        yield from self.chain.stream(
            {"input": question, "context": outcome.context_text()},
            session_config or config.session_config,
        )


if __name__ == '__main__':
    service = RagService()

    # 库内问题：应命中并正常回答
    print("--- 库内问题 ---")
    print(service.answer("我之前问了什么"))
    print(service.last_retrieval.debug)

    # 库外问题：应被阈值门控拦下
    print("--- 库外问题 ---")
    print(service.answer("今天北京的天气怎么样"))
    print(service.last_retrieval.debug)
