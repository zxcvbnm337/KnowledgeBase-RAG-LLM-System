"""分层对话记忆的离线测试。

用桩摘要器（stub）替代真实模型，因此**不消耗任何 API 额度、不需要 API Key**，
可以随时运行：

    python tests/test_memory_layering.py

覆盖点：
  1. 短期窗口与压缩触发时机
  2. 注入 prompt 的历史长度存在上界（核心不变式）
  3. 兼容旧版「纯消息列表」历史文件
  4. 摘要失败时保留全部原文（不丢历史）
  5. clear 与多会话隔离
"""

import json
import os
import shutil
import sys
import tempfile

# 允许从仓库任意位置执行本脚本
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage  # noqa: E402
from file_history_store import FileChatMessageHistory  # noqa: E402

_results = []


def check(label, cond, extra=""):
    _results.append(bool(cond))
    print(("  [PASS] " if cond else "  [FAIL] ") + label + (f"  {extra}" if extra else ""))


def stub_summarizer(previous, dialogue):
    """桩摘要器：把历史直接拼起来，便于断言「旧内容确实进了摘要」。"""
    flat = dialogue.replace("\n", " / ")
    return f"{previous} || {flat}" if previous else flat


def stub_failing(previous, dialogue):
    raise RuntimeError("模拟模型超时")


def section(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def round_of(h, text):
    h.add_messages([HumanMessage(content=text), AIMessage(content="好的")])


def test_window_and_trigger(tmp):
    section("测试 1：短期窗口与压缩触发 / 注入历史有上界")
    h = FileChatMessageHistory(
        "sess_a", storage_path=tmp,
        recent_messages=6, summary_trigger=10, summarizer=stub_summarizer,
    )
    check("summary_trigger 被校正为 10", h.summary_trigger == 10)

    for r in range(1, 8):
        round_of(h, f"第{r}轮：身高180体重75")
        s = h.stats()
        check(f"第{r}轮后磁盘原文 <= 阈值", s["recent_messages"] <= 10,
              f"实际 {s['recent_messages']}")

    st = h.stats()
    check("第 6 轮触发首次压缩", st["compressed_count"] == 1, f"次数 {st['compressed_count']}")
    check("长期摘要非空", st["summary_chars"] > 0)

    msgs = h.messages
    check("注入历史 = 摘要对(2) + 磁盘原文", len(msgs) == 2 + st["recent_messages"],
          f"实际 {len(msgs)}")
    check("首条不是 SystemMessage（角色交替合法）",
          not isinstance(msgs[0], SystemMessage))
    check("摘要承载在 Human 消息中", isinstance(msgs[0], HumanMessage))
    check("摘要里带早期事实", "第1轮" in str(msgs[0].content))

    # 核心不变式：聊到 25 轮，注入历史仍不超过 2 + 阈值
    peak = 0
    for r in range(8, 26):
        round_of(h, f"第{r}轮")
        peak = max(peak, len(h.messages))
    upper = 2 + h.summary_trigger
    print(f"    25 轮后：注入历史峰值 {peak} 条 / 上界 {upper} 条；"
          f"若不压缩应为 {2 * 25 + 1} 条")
    check(f"注入历史有上界（<= {upper}）", peak <= upper, f"峰值 {peak}")
    check("压缩被反复触发（摘要持续合并）", h.stats()["compressed_count"] >= 3,
          f"次数 {h.stats()['compressed_count']}")
    return h


def test_full_history_baseline(tmp):
    section("测试 2：对照「不压缩」的旧行为")
    h = FileChatMessageHistory(
        "sess_b", storage_path=tmp,
        recent_messages=6, summary_trigger=10, summarizer=None,
    )
    for r in range(1, 8):
        round_of(h, f"第{r}轮")
    check("摘要器为 None 时退化为全量保留（14 条）", len(h.messages) == 14,
          f"实际 {len(h.messages)}")


def test_legacy_format(tmp):
    section("测试 3：兼容旧版「纯消息列表」历史文件")
    legacy = os.path.join(tmp, "sess_legacy")
    with open(legacy, "w", encoding="utf-8") as f:
        json.dump([
            {"type": "human", "data": {"content": "旧格式提问", "type": "human"}},
            {"type": "ai", "data": {"content": "旧格式回答", "type": "ai"}},
        ], f)

    h = FileChatMessageHistory("sess_legacy", storage_path=tmp,
                               recent_messages=6, summary_trigger=10,
                               summarizer=stub_summarizer)
    check("旧文件可正常读取", len(h.messages) == 2, f"实际 {len(h.messages)}")
    h.add_messages([HumanMessage(content="迁移后的新提问")])
    check("旧记录迁移进新格式", h.stats()["recent_messages"] == 3)
    check("新文件落盘为 .json", os.path.exists(os.path.join(tmp, "sess_legacy.json")))


def test_summary_failure(tmp):
    section("测试 4：摘要失败时不丢历史")
    h = FileChatMessageHistory("sess_err", storage_path=tmp,
                               recent_messages=6, summary_trigger=10,
                               summarizer=stub_failing)
    for r in range(1, 8):
        round_of(h, f"第{r}轮")
    check("摘要失败时保留全部原文", len(h.messages) == 14, f"实际 {len(h.messages)}")


def test_clear_and_isolation(tmp):
    section("测试 5：clear 与多会话隔离")
    h1 = FileChatMessageHistory("sess_x", storage_path=tmp,
                                recent_messages=6, summary_trigger=10,
                                summarizer=stub_summarizer)
    h2 = FileChatMessageHistory("sess_y", storage_path=tmp,
                                recent_messages=6, summary_trigger=10,
                                summarizer=stub_summarizer)
    round_of(h1, "会话 X 的内容")
    round_of(h2, "会话 Y 的内容")
    check("两个会话互不干扰",
          "X" in str(h1.messages[0].content) and "Y" in str(h2.messages[0].content))

    h1.clear()
    check("clear 后历史清空", len(h1.messages) == 0)
    check("clear 后摘要清空", h1.stats()["summary_chars"] == 0)
    check("clear 不影响其他会话", len(h2.messages) == 2)

    sneaky = FileChatMessageHistory("../../etc/passwd", storage_path=tmp)
    check("session_id 中的路径分隔符被过滤",
          os.path.dirname(os.path.abspath(sneaky.file_path))
          == os.path.abspath(tmp))


def main():
    tmp = tempfile.mkdtemp(prefix="memtest_")
    try:
        test_window_and_trigger(tmp)
        test_full_history_baseline(tmp)
        test_legacy_format(tmp)
        test_summary_failure(tmp)
        test_clear_and_isolation(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    passed, total = sum(_results), len(_results)
    print()
    print("=" * 72)
    if passed == total:
        print(f"全部通过 ✅  {passed}/{total}")
    else:
        print(f"存在失败项 ❌  {passed}/{total}")
    print("=" * 72)
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
