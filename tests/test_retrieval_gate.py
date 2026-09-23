"""检索门控（召回 → 重排 → 阈值拒答）的离线测试。

**零依赖**：本脚本只 import 标准库与 `retrieval.py`，
不加载 langchain / chromadb / dashscope，也不需要 API Key：

    python tests/test_retrieval_gate.py

覆盖点：
  1. 距离 → 相似度的归一化与边界兜底
  2. 空查询 / 零召回 的拒答
  3. 阈值门控：全部被挡下时拒答，部分通过时计数正确
  4. final_k 截断
  5. 重排改变排序，且重排分数优先于向量相似度
  6. 重排器异常时优雅降级（退回向量顺序，不崩）
  7. 关闭重排时改判向量相似度阈值
  8. 注入 prompt 的参考资料格式
"""

import os
import sys

# 允许从仓库任意位置执行本脚本
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval import (  # noqa: E402
    DashScopeReranker,
    NoopReranker,
    RerankResult,
    Reranker,
    RetrievalPipeline,
    RetrievedChunk,
    distance_to_similarity,
)

_results = []


def check(label, cond, extra=""):
    _results.append(bool(cond))
    print(("  [PASS] " if cond else "  [FAIL] ") + label + (f"  {extra}" if extra else ""))


def section(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def make_search(pairs):
    """构造一个桩检索函数：pairs = [(内容, 距离), ...]，按距离升序返回。"""
    calls = []

    def _search(query, k):
        calls.append((query, k))
        chunks = [
            RetrievedChunk(content=text, metadata={"source": f"doc{i}.txt"},
                           vector_distance=dist)
            for i, (text, dist) in enumerate(pairs)
        ]
        chunks.sort(key=lambda c: c.vector_distance)
        return chunks[:k]

    _search.calls = calls
    return _search


class ReverseReranker(Reranker):
    """桩重排器：把顺序整个倒过来，便于断言「重排确实生效」。"""

    name = "reverse"

    def rerank(self, query, chunks, top_k):
        ordered = list(reversed(list(chunks)))
        for i, c in enumerate(ordered):
            c.rerank_score = float(len(ordered) - i) / 10.0
        return RerankResult(ordered[:top_k])


class BoomReranker(Reranker):
    """桩重排器：模拟网络异常 / SDK 变更。"""

    name = "boom"

    def rerank(self, query, chunks, top_k):
        raise RuntimeError("模拟重排服务超时")


# ---------------------------------------------------------------- 测试


def test_distance_to_similarity():
    section("测试 1：距离 → 相似度的归一化与边界兜底")
    check("距离 0 → 相似度 1.0", abs(distance_to_similarity(0) - 1.0) < 1e-9)
    check("距离 1 → 相似度 0.5", abs(distance_to_similarity(1) - 0.5) < 1e-9)
    check("单调递减（距离越大相似度越小）",
          distance_to_similarity(0.5) > distance_to_similarity(1.5) > distance_to_similarity(3.0))
    check("负距离不越界", distance_to_similarity(-5) == 1.0)
    check("NaN 兜底为 0", distance_to_similarity(float("nan")) == 0.0)
    check("非数值输入兜底为 0", distance_to_similarity(None) == 0.0)
    check("L2 距离 > 1 时相似度仍为正",
          distance_to_similarity(4.0) > 0, f"实际 {distance_to_similarity(4.0):.4f}")


def test_empty_and_no_candidate():
    section("测试 2：空查询 / 零召回 的拒答")
    pipe = RetrievalPipeline(search_fn=make_search([("内容", 0.1)]))
    out = pipe.retrieve("   ")
    check("空查询被拒答", out.refused)
    check("阶段标记为 empty_query", out.debug.get("stage") == "empty_query")
    check("空查询不触发检索", pipe.search_fn.calls == [])

    pipe2 = RetrievalPipeline(search_fn=make_search([]))
    out2 = pipe2.retrieve("随便问问")
    check("零召回被拒答", out2.refused)
    check("阶段标记为 no_candidate", out2.debug.get("stage") == "no_candidate")
    check("拒答时不返回片段", out2.chunks == [])


def test_threshold_gate():
    section("测试 3：阈值门控（拒答 / 部分通过 / 计数）")
    # 距离 9.0 → 相似度 0.1，低于默认向量阈值 0.45
    far = [("无关内容A", 9.0), ("无关内容B", 12.0)]
    out = RetrievalPipeline(search_fn=make_search(far)).retrieve("库外问题")
    check("全部低于阈值 → 拒答", out.refused)
    check("阶段标记为 below_threshold", out.debug.get("stage") == "below_threshold")
    check("记录了最高分", out.debug["best_score"] > 0, f"best={out.debug['best_score']:.4f}")

    # 距离 0.5/0.9 → 相似度 0.667/0.526（均过 0.45）；距离 6.0 → 0.143（被挡）
    mixed = [("高度相关", 0.5), ("一般相关", 0.9), ("不相关", 6.0)]
    out2 = RetrievalPipeline(search_fn=make_search(mixed)).retrieve("库内问题")
    check("部分通过时不予拒答", not out2.refused)
    check("只保留过阈值的片段", len(out2.chunks) == 2, f"实际 {len(out2.chunks)}")
    check("被阈值挡下的计数正确", out2.debug["dropped_by_threshold"] == 1,
          f"实际 {out2.debug['dropped_by_threshold']}")
    check("判定依据为向量相似度", out2.debug["score_source"] == "vector_similarity")


def test_final_k_truncation():
    section("测试 4：final_k 截断与 top_k 透传")
    pairs = [(f"片段{i}", 0.1 * i + 0.1) for i in range(6)]
    search = make_search(pairs)
    out = RetrievalPipeline(search_fn=search, top_k=5, final_k=2).retrieve("查询")
    check("top_k 透传给检索函数", search.calls and search.calls[0][1] == 5,
          f"实际 {search.calls}")
    check("候选被 top_k 限制为 5", out.debug["candidates"] == 5,
          f"实际 {out.debug['candidates']}")
    check("最终注入被 final_k 截断为 2", len(out.chunks) == 2, f"实际 {len(out.chunks)}")


def test_rerank_changes_order():
    section("测试 5：重排生效，且重排分数优先于向量相似度")
    pairs = [("向量第一名", 0.2), ("向量第二名", 0.4), ("向量第三名", 0.6)]
    out = RetrievalPipeline(
        search_fn=make_search(pairs), reranker=ReverseReranker(), final_k=3,
    ).retrieve("查询")

    check("重排确实改变了顺序", out.chunks[0].content == "向量第三名",
          f"实际 {out.chunks[0].content}")
    check("片段已被打上重排分数", out.chunks[0].rerank_score is not None)
    check("判定依据切换为重排分数", out.debug["score_source"] == "rerank")
    check("阈值切换为重排阈值（0.05）", out.debug["threshold"] == 0.05,
          f"实际 {out.debug['threshold']}")
    check("重排未降级", out.debug["rerank_degraded"] is False)

    # 关键：重排分数必须压过向量相似度，否则门控会拿错分数
    chunk = out.chunks[0]
    check("score 取重排分数而非向量相似度",
          abs(chunk.score - chunk.rerank_score) < 1e-9,
          f"score={chunk.score} rerank={chunk.rerank_score} "
          f"vec={chunk.vector_similarity:.4f}")


def test_rerank_degradation():
    section("测试 6：重排器异常时优雅降级（管线双重保险）")
    # BoomReranker 直接抛异常，违反「不抛异常」约定 —— 管线必须自己兜住
    pairs = [("第一名", 0.2), ("第二名", 0.4)]
    out = RetrievalPipeline(
        search_fn=make_search(pairs), reranker=BoomReranker(), final_k=2,
    ).retrieve("查询")
    check("降级后仍然给出结果（不崩）", not out.refused)
    check("降级标记为 True", out.debug["rerank_degraded"] is True)
    check("降级原因被记录", "RuntimeError" in out.debug["rerank_error"],
          out.debug["rerank_error"])
    check("降级后退回向量顺序", out.chunks[0].content == "第一名",
          f"实际 {out.chunks[0].content}")
    check("降级后无重排分数", out.chunks[0].rerank_score is None)
    check("降级后判定依据退回向量相似度", out.debug["score_source"] == "vector_similarity")


def test_dashscope_reranker_never_raises():
    section("测试 7：真实重排器在无 Key / 无网络时也不抛异常")
    rr = DashScopeReranker(api_key=None)
    chunks = [RetrievedChunk(content="甲", vector_distance=0.1),
              RetrievedChunk(content="乙", vector_distance=0.2)]
    # 不设 api_key 时 dashscope 可能直接报鉴权错误 —— 必须被吞掉并降级
    res = rr.rerank("查询", chunks, 2)
    check("返回 RerankResult 而非抛异常", isinstance(res, RerankResult))
    check("返回片段数不超过 top_k", len(res.chunks) <= 2)
    print(f"    降级状态 degraded={res.degraded}  error={res.error[:70] or '（无）'}")


def test_noop_and_disabled_rerank():
    section("测试 8：关闭重排 / Noop 占位重排")
    pairs = [("A", 0.5), ("B", 0.9)]
    out = RetrievalPipeline(search_fn=make_search(pairs), reranker=None).retrieve("查询")
    check("关闭重排时按向量相似度判定", out.debug["score_source"] == "vector_similarity")
    check("rerank_enabled 标记为 False", out.debug["rerank_enabled"] is False)

    out2 = RetrievalPipeline(search_fn=make_search(pairs),
                             reranker=NoopReranker()).retrieve("查询")
    check("Noop 重排不改变顺序", out2.chunks[0].content == "A")
    check("Noop 不给片段打分", out2.chunks[0].rerank_score is None,
          "因此判定依据仍为向量相似度")


def test_context_text():
    section("测试 9：注入 prompt 的参考资料格式")
    out = RetrievalPipeline(search_fn=make_search([("尺码表内容", 0.2)])).retrieve("查询")
    ctx = out.context_text()
    check("包含『文档片段』标记", "文档片段：尺码表内容" in ctx)
    check("包含『文档元数据』标记", "文档元数据：" in ctx)
    check("保留了来源信息", "doc0.txt" in ctx)

    empty = RetrievalPipeline(search_fn=make_search([])).retrieve("查询")
    check("无片段时返回既有约定的占位文案",
          empty.context_text() == "无相关参考资料", empty.context_text())


def main():
    test_distance_to_similarity()
    test_empty_and_no_candidate()
    test_threshold_gate()
    test_final_k_truncation()
    test_rerank_changes_order()
    test_rerank_degradation()
    test_dashscope_reranker_never_raises()
    test_noop_and_disabled_rerank()
    test_context_text()

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
