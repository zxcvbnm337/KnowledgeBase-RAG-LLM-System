"""检索链路的「召回 → 重排 → 阈值门控」三段式实现。

设计原则（与本项目既有风格保持一致）：

1. **纯逻辑与外部依赖解耦**：本模块顶层不 import langchain / dashscope，
   因此 `tests/test_retrieval_gate.py` 可以在**零依赖**环境下直接运行。
2. **优雅降级**：重排失败不抛异常，退回向量召回顺序并标记 `degraded`，
   与小程序项目「云挂了页面不崩」是同一套思路。
3. **可解释**：每个环节的中间分数都保留在 `RetrievalOutcome.debug` 中，
   供页面展示与离线评测消费——检索质量必须能被度量，否则调参只能靠感觉。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

# ==================== 数据结构 ====================


@dataclass
class RetrievedChunk:
    """一个检索候选片段，同时承载逐个环节的分数。"""

    content: str
    metadata: dict = field(default_factory=dict)
    vector_distance: Optional[float] = None      # Chroma 返回的距离，越小越近
    vector_similarity: Optional[float] = None    # 归一化后的相似度，越大越相关
    rerank_score: Optional[float] = None         # 重排模型给出的相关性分数

    @property
    def score(self) -> float:
        """当前生效的「越大越相关」分数。

        重排可用时以重排分数为准（实测区分度远高于向量相似度）；
        重排不可用时退化为归一化相似度。
        """
        if self.rerank_score is not None:
            return float(self.rerank_score)
        if self.vector_similarity is not None:
            return float(self.vector_similarity)
        return 0.0

    @property
    def score_source(self) -> str:
        return "rerank" if self.rerank_score is not None else "vector_similarity"

    def preview(self, width: int = 60) -> str:
        flat = " ".join(str(self.content).split())
        return flat[:width] + ("…" if len(flat) > width else "")


# ==================== 分数归一化 ====================


def distance_to_similarity(distance: float) -> float:
    """Chroma 默认 L2 距离 → (0, 1] 的相似度。

    用 `1 / (1 + d)` 而不是 `1 - d`：距离可能大于 1（L2 未归一化），
    直接相减会得到负数，破坏「越大越相关」的单调语义。
    """
    try:
        d = float(distance)
    except (TypeError, ValueError):
        return 0.0
    if d != d:  # NaN
        return 0.0
    if d < 0:
        d = 0.0
    return 1.0 / (1.0 + d)


# ==================== 重排器 ====================


@dataclass
class RerankResult:
    chunks: list
    degraded: bool = False       # True 表示重排不可用、已退回向量顺序
    error: str = ""


class Reranker(object):
    """重排器接口。实现方必须**永不抛异常**，失败时返回 degraded 结果。"""

    name = "base"

    def rerank(self, query: str, chunks: Sequence[RetrievedChunk], top_k: int) -> RerankResult:
        raise NotImplementedError


class NoopReranker(Reranker):
    """占位重排器：仅做截断，用于关闭重排时做 A/B 对照。"""

    name = "noop"

    def rerank(self, query, chunks, top_k):
        return RerankResult(chunks=list(chunks)[:top_k])


class DashScopeReranker(Reranker):
    """基于 DashScope `gte-rerank-v2` 的交叉编码重排。

    注意：dashscope 是**延迟导入**的，未安装或未配置 Key 时本模块依然可导入，
    只是调用时返回 degraded —— 保证离线测试不依赖任何外部 SDK。
    """

    def __init__(self, model: str = "gte-rerank-v2", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key
        self.name = model

    def rerank(self, query, chunks, top_k):
        chunks = list(chunks)
        if not chunks:
            return RerankResult(chunks=[])
        try:
            import dashscope  # 延迟导入
        except ImportError as e:
            return RerankResult(chunks[:top_k], degraded=True,
                                error=f"dashscope 未安装：{e}")

        try:
            if self.api_key:
                dashscope.api_key = self.api_key
            resp = dashscope.TextReRank.call(
                model=self.model,
                query=query,
                documents=[c.content for c in chunks],
                top_n=min(top_k, len(chunks)),
                return_documents=False,
            )
            if getattr(resp, "status_code", 200) != 200:
                return RerankResult(chunks[:top_k], degraded=True,
                                    error=f"HTTP {getattr(resp, 'status_code', '?')}: "
                                          f"{getattr(resp, 'message', '')}")

            results = (resp.output or {}).get("results") or []
            if not results:
                return RerankResult(chunks[:top_k], degraded=True, error="重排返回空结果")

            ordered = []
            for item in results:
                idx = item.get("index")
                if not isinstance(idx, int) or not (0 <= idx < len(chunks)):
                    continue
                chunk = chunks[idx]
                chunk.rerank_score = float(item.get("relevance_score") or 0.0)
                ordered.append(chunk)
            if not ordered:
                return RerankResult(chunks[:top_k], degraded=True, error="重排索引越界")

            ordered.sort(key=lambda c: c.score, reverse=True)
            return RerankResult(ordered[:top_k])
        except Exception as e:  # 网络、鉴权、限流、SDK 变更 —— 一律降级，不影响主链路
            return RerankResult(chunks[:top_k], degraded=True,
                                error=f"{type(e).__name__}: {e}")


# ==================== 检索结果 ====================


@dataclass
class RetrievalOutcome:
    query: str
    chunks: list = field(default_factory=list)   # 最终注入 prompt 的片段
    refused: bool = False
    reason: str = ""
    debug: dict = field(default_factory=dict)

    def context_text(self) -> str:
        """拼成注入 prompt 的参考资料文本，保持与改造前一致的格式。"""
        if not self.chunks:
            return "无相关参考资料"
        parts = []
        for c in self.chunks:
            parts.append(f"文档片段：{c.content}\n文档元数据：{c.metadata}\n")
        return "\n".join(parts)


# ==================== 检索引擎 ====================


class RetrievalPipeline(object):
    """召回 → 重排 → 阈值门控。

    :param search_fn: `(query, k) -> list[RetrievedChunk]`，注入式设计，
        离线测试传桩函数即可，不需要向量库。
    :param reranker: 重排器；传 None 表示关闭重排。
    :param rerank_threshold: 重排分数阈值（启用重排时生效）。
    :param vector_similarity_threshold: 归一化相似度阈值（未启用重排时生效）。
    """

    def __init__(
        self,
        search_fn: Callable[[str, int], list],
        reranker: Optional[Reranker] = None,
        top_k: int = 8,
        final_k: int = 3,
        rerank_threshold: float = 0.05,
        vector_similarity_threshold: float = 0.45,
        refusal_text: str = "抱歉，这个问题不在我的知识库范围内。",
    ):
        self.search_fn = search_fn
        self.reranker = reranker
        self.top_k = max(1, int(top_k))
        self.final_k = max(1, int(final_k))
        self.rerank_threshold = float(rerank_threshold)
        self.vector_similarity_threshold = float(vector_similarity_threshold)
        self.refusal_text = refusal_text

    def retrieve(self, query: str) -> RetrievalOutcome:
        query = (query or "").strip()
        if not query:
            return RetrievalOutcome(query=query, refused=True, reason="空查询",
                                    debug={"stage": "empty_query"})

        candidates = list(self.search_fn(query, self.top_k) or [])
        for c in candidates:
            if c.vector_similarity is None and c.vector_distance is not None:
                c.vector_similarity = distance_to_similarity(c.vector_distance)

        debug = {
            "stage": "ok",
            "candidates": len(candidates),
            "rerank_enabled": self.reranker is not None,
            "rerank_degraded": False,
            "rerank_error": "",
            "threshold": None,
            "best_score": 0.0,
            "dropped_by_threshold": 0,
        }

        if not candidates:
            debug["stage"] = "no_candidate"
            return RetrievalOutcome(query=query, refused=True, reason="知识库中没有召回任何片段",
                                    debug=debug)

        # ---- 重排 ----
        if self.reranker is not None:
            try:
                rr = self.reranker.rerank(query, candidates, self.top_k)
            except Exception as e:
                # 双重保险：即便某个重排实现违反了「不抛异常」的约定，
                # 也不能把整条问答链路带崩。
                rr = RerankResult(list(candidates)[:self.top_k], degraded=True,
                                  error=f"{type(e).__name__}: {e}")
            ordered = rr.chunks
            debug["rerank_degraded"] = rr.degraded
            debug["rerank_error"] = rr.error
        else:
            ordered = list(candidates)

        # ---- 阈值门控 ----
        used_rerank = any(c.rerank_score is not None for c in ordered)
        threshold = self.rerank_threshold if used_rerank else self.vector_similarity_threshold
        debug["threshold"] = threshold
        debug["score_source"] = "rerank" if used_rerank else "vector_similarity"

        passed = [c for c in ordered if c.score >= threshold]
        debug["best_score"] = ordered[0].score if ordered else 0.0
        debug["dropped_by_threshold"] = len(ordered) - len(passed)

        if not passed:
            debug["stage"] = "below_threshold"
            return RetrievalOutcome(
                query=query, refused=True,
                reason=f"最高相关度 {debug['best_score']:.4f} 低于阈值 {threshold}",
                debug=debug,
            )

        kept = passed[:self.final_k]
        debug["kept"] = [{"score": round(c.score, 4), "preview": c.preview(40)} for c in kept]
        debug["score_source"] = kept[0].score_source
        return RetrievalOutcome(query=query, chunks=kept, debug=debug)
