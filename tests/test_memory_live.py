"""分层对话记忆的真实联调测试（会调用模型，消耗 API 额度）。

需要先在项目根目录配置好 `.env` 中的 `DASHSCOPE_API_KEY`，且本机向量库已有数据。

    python tests/test_memory_live.py

验证三件事：
  1. 摘要由真实模型生成，长度受控（不随轮次线性增长）；
  2. 第 1 轮说过的关键事实，在经历多次压缩后仍能被追问出来；
  3. 注入 prompt 的历史长度始终有上界。
"""

import os
import re
import sys
import time

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 项目内大量使用相对路径（./chroma_db、./chat_history），切到根目录保证一致
os.chdir(PROJ)
sys.path.insert(0, PROJ)

from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(PROJ, ".env"))

import config_data as config  # noqa: E402
import rag as rag_module  # noqa: E402
from file_history_store import FileChatMessageHistory  # noqa: E402
from langchain_core.messages import SystemMessage  # noqa: E402

rag_module.print_prompt = lambda p: p  # 关掉完整 prompt 刷屏

SESSION = "__memory_live_check__"
QUESTION = {"configurable": {"session_id": SESSION}}

ROUNDS = [
    "我身高180厘米，体重75公斤",
    "那我上衣该选什么尺码？",
    "这件衣服平时怎么保养？",
    "深色和浅色可以一起洗吗？",
    "灰色适合我吗？",
    "夏天穿什么颜色比较凉快？",
    "那冬天呢？",
    "好的，我适合的颜色再确认一下",
    "我最开始告诉你的身高和体重是多少？",   # 第 1 轮的信息此时只存在于摘要里
    "收到，谢谢",
    "那我记住了",
    "再问一次，我的身高体重是多少？",       # 经历第二次压缩后再次回忆
]


def read_memory():
    h = FileChatMessageHistory(SESSION, storage_path=config.chat_history_dir)
    msgs = h.messages
    kinds = [
        "摘要" if str(m.content).startswith("（以下是更早对话的摘要") else m.type
        for m in msgs
    ]
    return h, msgs, kinds


def main():
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("缺少 DASHSCOPE_API_KEY，请先配置项目根目录的 .env")
        return 2

    stale = os.path.join(PROJ, "chat_history", SESSION + ".json")
    if os.path.exists(stale):
        os.remove(stale)

    print("=" * 74)
    print("初始化 RagService（加载向量库 + 两个模型）…")
    svc = rag_module.RagService()
    print("就绪\n")

    answers = []
    started = time.time()
    for idx, question in enumerate(ROUNDS, 1):
        answer = re.sub(r"\s+", " ", str(svc.chain.invoke({"input": question}, QUESTION))).strip()
        answers.append(answer)
        stats = svc.memory_stats(SESSION)
        _, msgs, kinds = read_memory()
        print(f"[第{idx:>2}轮] 问：{question}")
        print(f"         答：{answer[:80]}{'…' if len(answer) > 80 else ''}")
        print(f"         记忆 -> 磁盘 {stats['recent_messages']} 条 | "
              f"摘要 {stats['summary_chars']} 字 | 压缩 {stats['compressed_count']} 次 | "
              f"注入 {len(msgs)} 条 {kinds}\n")

    h, msgs, _ = read_memory()
    stats = svc.memory_stats(SESSION)
    summary = h._data.get("summary", "")
    baseline = len(ROUNDS) * 2 + 1

    print("=" * 74)
    print("最终状态")
    print("=" * 74)
    print(f"  对话轮次           : {len(ROUNDS)}")
    print(f"  磁盘保留原文       : {stats['recent_messages']} 条")
    print(f"  长期摘要长度       : {stats['summary_chars']} 字")
    print(f"  累计压缩次数       : {stats['compressed_count']}")
    print(f"  注入 prompt 历史   : {len(msgs)} 条（不压缩应为 {baseline} 条）")
    print(f"  总耗时             : {time.time() - started:.0f} 秒")
    print("\n--- 长期摘要全文 ---")
    print(summary)

    print()
    print("=" * 74)
    print("关键断言")
    print("=" * 74)
    checks = [
        ("至少发生两次压缩（验证摘要能持续合并）", stats["compressed_count"] >= 2),
        ("摘要长度受控（0 < 摘要 < 1200 字）", 0 < stats["summary_chars"] < 1200),
        ("注入历史少于全量", len(msgs) < baseline),
        ("注入历史有上界",
         len(msgs) <= 2 + config.memory_summary_trigger),
        ("首条不是 SystemMessage（角色交替合法）",
         not isinstance(msgs[0], SystemMessage)),
        ("摘要保留了身高体重", "180" in summary and "75" in summary),
        ("第 9 轮正确回忆出身高体重（跨越首次压缩）",
         "180" in answers[8] and "75" in answers[8]),
        ("第 12 轮正确回忆出身高体重（跨越二次压缩）",
         "180" in answers[11] and "75" in answers[11]),
    ]
    ok = True
    for label, cond in checks:
        print(("  [PASS] " if cond else "  [FAIL] ") + label)
        ok = ok and cond

    # 清理本次测试产生的历史文件
    for path in (stale, os.path.join(PROJ, "chat_history", SESSION)):
        if os.path.exists(path):
            os.remove(path)

    print()
    print("真实联调" + ("通过 ✅" if ok else "存在失败项 ❌"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
