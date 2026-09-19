"""会话历史存储：带「短期原文 + 长期摘要」的分层记忆。

为什么要分层
------------
原始实现每次 add_messages 都把整个历史文件读出来、追加、再整个写回，
消息只增不减。对话聊到几十轮之后，注入 prompt 的历史会无限变长：

  * token 成本随轮次线性上涨；
  * 最终一定会超出模型上下文窗口，直接报错中断对话。

分层之后
--------
  * 短期记忆：最近 memory_recent_messages 条消息保留原文，
    保证「上面那个」「它」这类指代能被正确理解；
  * 长期记忆：更早的消息交给 LLM 压缩成一段摘要，只保留关键信息。

于是不管聊多少轮，注入 prompt 的历史长度都稳定在一个可控范围内，
而早期聊过的关键事实又不会丢——这才是「记忆」而不是「截断」。
"""

import json
import os
import re
from typing import Callable, Optional, Sequence

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    message_to_dict,
    messages_from_dict,
)

import config_data as config


class FileChatMessageHistory(BaseChatMessageHistory):
    """把单个会话的历史按「摘要 + 最近原文」两层落盘到本地 JSON 文件。"""

    STORAGE_VERSION = 2

    ROLE_LABELS = {
        "human": "用户",
        "ai": "助手",
        "system": "系统",
        "tool": "工具",
        "function": "函数",
    }

    # session_id 会拼进文件名，先过滤掉路径分隔符等危险字符
    _UNSAFE_CHARS = re.compile(r"[^0-9A-Za-z_.\-]")

    def __init__(
        self,
        session_id: str,
        storage_path: str = config.chat_history_dir,
        recent_messages: int = config.memory_recent_messages,
        summary_trigger: int = config.memory_summary_trigger,
        summarizer: Optional[Callable[[str, str], str]] = None,
    ):
        self.session_id = session_id
        self.storage_path = storage_path
        self.recent_messages = recent_messages
        # 触发点必须大于短期窗口，否则会反复压缩同一批消息且不收敛
        self.summary_trigger = max(summary_trigger, recent_messages + 2)
        # 摘要函数签名：(已有摘要, 新对话文本) -> 新摘要；为 None 时退化为「全量保留」
        self.summarizer = summarizer

        safe_id = self._UNSAFE_CHARS.sub("_", session_id) or "default"
        os.makedirs(self.storage_path, exist_ok=True)
        self.file_path = os.path.join(self.storage_path, f"{safe_id}.json")
        # 旧版本把文件存成无扩展名的 ./chat_history/<session_id>
        self._legacy_path = os.path.join(self.storage_path, session_id)

        self._data = self._load()

    # ------------------------------------------------------------------ 读写
    def _load(self) -> dict:
        raw = None
        for path in (self.file_path, self._legacy_path):
            if not os.path.exists(path):
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                break
            except (json.JSONDecodeError, OSError):
                # 文件被手动清空或写坏时不要让整个对话崩掉
                continue
        return self._normalize(raw)

    @classmethod
    def _normalize(cls, raw) -> dict:
        """兼容三种历史数据：空 / 旧版纯列表 / 新版字典。"""
        if isinstance(raw, list):
            # v1 格式：文件内容就是一个消息列表，没有摘要概念
            return {
                "version": cls.STORAGE_VERSION,
                "summary": "",
                "messages": raw,
                "compressed_count": 0,
            }
        if isinstance(raw, dict):
            return {
                "version": cls.STORAGE_VERSION,
                "summary": raw.get("summary") or "",
                "messages": raw.get("messages") or [],
                "compressed_count": raw.get("compressed_count") or 0,
            }
        return {
            "version": cls.STORAGE_VERSION,
            "summary": "",
            "messages": [],
            "compressed_count": 0,
        }

    def _save(self) -> None:
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------- 框架约定的三个成员
    @property
    def messages(self) -> list[BaseMessage]:
        """返回真正注入 prompt 的历史：长期摘要 + 最近的短期原文。

        注意这里是分层后的视图，不是磁盘上的全部原文；
        框架正是通过这个属性把记忆喂给模型的。
        """
        history: list[BaseMessage] = []

        summary = (self._data.get("summary") or "").strip()
        if summary:
            # 用「用户补充说明 + 助手确认」承载摘要，而不是插 SystemMessage：
            # 角色交替关系保持合法，换任何对话模型都不会因角色顺序报错。
            history.append(
                HumanMessage(content=f"（以下是更早对话的摘要，供你参考）\n{summary}")
            )
            history.append(AIMessage(content="好的，我已了解之前的对话背景。"))

        history.extend(messages_from_dict(self._data.get("messages", [])))
        return history

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        """追加消息，并在超过触发点时把更早的内容压缩进摘要。"""
        self._data.setdefault("messages", []).extend(
            message_to_dict(message) for message in messages
        )
        self._compress_if_needed()
        self._save()

    def clear(self) -> None:
        self._data = {
            "version": self.STORAGE_VERSION,
            "summary": "",
            "messages": [],
            "compressed_count": 0,
        }
        self._save()

    # ------------------------------------------------------------- 压缩逻辑
    def _compress_if_needed(self) -> None:
        messages = self._data.get("messages", [])
        if self.summarizer is None or len(messages) <= self.summary_trigger:
            return

        keep = self.recent_messages
        # 按「用户-助手」成对的方式切分，避免短期窗口以助手发言开头
        if (len(messages) - keep) % 2:
            keep += 1
        older, recent = messages[:-keep], messages[-keep:]

        try:
            summary = self.summarizer(
                self._data.get("summary", ""), self._format_dialogue(older)
            )
        except Exception as exc:
            # 压缩失败就保留全部原文：宁可多花 token，也不能把历史丢掉
            print(f"[记忆分层] 摘要生成失败，本轮保留完整历史：{exc}")
            return

        if not summary or not summary.strip():
            print("[记忆分层] 摘要返回为空，本轮保留完整历史")
            return

        self._data["summary"] = summary.strip()
        self._data["messages"] = recent
        self._data["compressed_count"] = self._data.get("compressed_count", 0) + 1
        print(
            f"[记忆分层] 第 {self._data['compressed_count']} 次压缩："
            f"{len(older)} 条消息 -> 摘要 {len(self._data['summary'])} 字，"
            f"短期窗口保留 {len(recent)} 条"
        )

    @classmethod
    def _format_dialogue(cls, messages: list[dict]) -> str:
        """把消息字典转成带角色标签的纯文本，供摘要模型阅读。"""
        lines = []
        for item in messages:
            try:
                message = messages_from_dict([item])[0]
            except Exception:
                continue
            content = (
                message.content
                if isinstance(message.content, str)
                else str(message.content)
            )
            label = cls.ROLE_LABELS.get(message.type, message.type)
            lines.append(f"{label}：{content}")
        return "\n".join(lines)

    # ----------------------------------------------------------------- 观测
    def stats(self) -> dict:
        """给界面/调试用：一眼看出分层记忆当前的工作状态。"""
        return {
            "summary_chars": len((self._data.get("summary") or "").strip()),
            "recent_messages": len(self._data.get("messages", [])),
            "compressed_count": self._data.get("compressed_count", 0),
            "summary_trigger": self.summary_trigger,
            "raw_message_limit": self.recent_messages,
        }


def get_history(session_id: str) -> FileChatMessageHistory:
    """不带摘要能力的兼容入口（保留给直接引用本模块的场景）。"""
    return FileChatMessageHistory(session_id, storage_path=config.chat_history_dir)
