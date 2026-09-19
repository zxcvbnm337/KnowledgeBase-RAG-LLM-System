from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory, RunnableLambda
from file_history_store import FileChatMessageHistory
from vector_stores import VectorStoreService
from langchain_community.embeddings import DashScopeEmbeddings
import config_data as config
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models.tongyi import ChatTongyi


def print_prompt(prompt):
    print("="*20)
    print(prompt.to_string())
    print("="*20)

    return prompt


class RagService(object):
    def __init__(self):

        self.vector_service = VectorStoreService(
            embedding=DashScopeEmbeddings(model=config.embedding_model_name)
        )

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
        """获取最终的执行链"""

        retriever = self.vector_service.get_retriever()

        def format_document(docs: list[Document]):
            if not docs:
                return "无相关参考资料"

            formatted_str = ""
            for doc in docs:
                formatted_str += f"文档片段：{doc.page_content}\n文档元数据：{doc.metadata}\n\n"

            return formatted_str

        def format_for_retriever(value: dict)->str:

            return value["input"]

        def format_for_prompt_template(value):
            # {input, context, history}
            new_value = {}
            new_value["input"] = value["input"]["input"]
            new_value["context"] = value["context"]
            new_value["history"] = value["input"]["history"]
            return new_value


        chain = (
            {
                "input": RunnablePassthrough(),
                "context": RunnableLambda(format_for_retriever) | retriever | format_document
            }| RunnableLambda(format_for_prompt_template) |self.prompt_template | print_prompt |self.chat_model | StrOutputParser()
        )

        conversation_chain = RunnableWithMessageHistory(       # 增强的链
            chain,
            self._get_history,
            input_messages_key="input",
            history_messages_key="history",
        )

        return conversation_chain


if __name__ == '__main__':
    # session id 配置
    session_config ={
        "configurable":{
            "session_id":"user_001",
        }
    }
    res = RagService().chain.invoke({"input":"我之前问了什么"},session_config)
    print(res)
