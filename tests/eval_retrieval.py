"""检索效果评测：量化「加不加重排」的差别。

与 test_retrieval_gate.py 的分工：
    - test_retrieval_gate.py —— **零依赖离线**断言，验证门控逻辑本身正确（快、免费、随时跑）
    - eval_retrieval.py      —— **需要 API Key** 的真实评测，衡量检索质量好坏（慢、花钱、按需跑）

为什么要有这一层：
    调阈值、换重排模型、改 chunk_size —— 每一次调整都需要一个可比的基线，
    否则「感觉好像准了一点」会持续累积成不可控的技术债。

运行：
    python tests/eval_retrieval.py

指标口径：
    Hit@1 / Hit@3 —— 前 1 / 前 3 条候选中是否包含标注答案所在片段
    MRR           —— 首个正确片段的排名倒数均值（1.0 表示永远排第一）
    库外准确率     —— 库外问题被正确拒答的比例（幻觉的第一道闸门）
    库内误拒率     —— 库内问题被错误拒答的比例（拒答太狠会伤可用性）
"""

import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

import config_data as config  # noqa: E402
from retrieval import (  # noqa: E402
    DashScopeReranker,
    RetrievalPipeline,
    distance_to_similarity,
)
from vector_stores import VectorStoreService  # noqa: E402

# ---------------- 评测集 ----------------
# 答案用「必须出现在该片段中的关键词」标注，避免把标注和 chunk 切分结果耦合。
# 关键词都选了在各片段中互斥的词，避免「随便命中哪条都算对」。
IN_DOMAIN_CASES = [
    ("身高180体重75穿什么尺码", "尺码"),
    ("165cm 55公斤 选哪个码数", "尺码"),
    ("白色T恤怎么洗", "纯棉"),
    ("雪纺连衣裙洗涤要注意什么", "雪纺"),
    ("加绒牛仔裤能机洗吗", "加绒"),
    ("皮肤偏黄适合穿什么颜色", "肤色"),
    ("颜色搭配有什么禁忌", "禁忌"),
]

OUT_OF_DOMAIN_CASES = [
    "今天北京的天气怎么样",
    "帮我写一段 Python 快速排序",
    "你们公司的退货政策是什么",
    "帮我订一张去上海的机票",
]


class CachedSearch(object):
    """给检索函数加一层缓存。

    一次评测里，同一个问题会被「算排序」和「跑门控」各取一次候选；
    缓存后只花一次嵌入调用。返回浅拷贝，避免重排器写入的分数污染缓存。
    """

    def __init__(self, fn):
        self.fn = fn
        self.cache = {}
        self.calls = 0

    def __call__(self, query, k):
        key = (query, k)
        if key not in self.cache:
            self.cache[key] = self.fn(query, k)
            self.calls += 1
        return [copy.copy(c) for c in self.cache[key]]


def evaluate(name, search, reranker, top_k, threshold_kwargs):
    """跑一遍评测集。空行分隔，输出逐题明细与汇总。"""
    pipeline = RetrievalPipeline(
        search_fn=search, reranker=reranker,
        top_k=top_k, final_k=config.retrieval_final_k,
        refusal_text=config.refusal_text, **threshold_kwargs,
    )

    print()
    print("=" * 82)
    print(f"配置 {name}")
    print("=" * 82)
    print(f"{'问题':<28}{'期望':<8}{'排名':<6}{'最高分':<10}{'结果'}")
    print("-" * 82)

    hits1 = hits3 = 0
    mrr_sum = 0.0
    false_refusal = 0
    in_top_scores = []

    for question, keyword in IN_DOMAIN_CASES:
        ordered = _ranked_candidates(search, reranker, question, top_k)
        rank = next((i + 1 for i, c in enumerate(ordered) if keyword in c.content), 0)
        outcome = pipeline.retrieve(question)

        hits1 += 1 if rank == 1 else 0
        hits3 += 1 if 1 <= rank <= 3 else 0
        mrr_sum += (1.0 / rank) if rank else 0.0
        false_refusal += 1 if outcome.refused else 0
        if ordered:
            in_top_scores.append(ordered[0].score)

        top = f"{ordered[0].score:.4f}" if ordered else "-"
        print(f"{question:<28}{keyword:<8}{rank or '-':<6}{top:<10}"
              f"{'误拒 ❌' if outcome.refused else '正常'}")

    refused_out = 0
    out_top_scores = []
    for question in OUT_OF_DOMAIN_CASES:
        ordered = _ranked_candidates(search, reranker, question, top_k)
        outcome = pipeline.retrieve(question)
        refused_out += 1 if outcome.refused else 0
        if ordered:
            out_top_scores.append(ordered[0].score)
        top = f"{ordered[0].score:.4f}" if ordered else "-"
        print(f"{question:<28}{'（库外）':<8}{'-':<6}{top:<10}"
              f"{'拒答 ✅' if outcome.refused else '未拒答 ❌'}")

    n = len(IN_DOMAIN_CASES)
    m = len(OUT_OF_DOMAIN_CASES)

    # 判定间隔：库内问题的最低分 ÷ 库外问题的最高分。
    # 这是「阈值能否可靠工作」的直接度量——比值越接近 1，阈值就越难定，
    # 迟早会掉进「拒答太狠」或「什么都答」的二选一。
    margin = (min(in_top_scores) / max(out_top_scores)) if (in_top_scores and out_top_scores
                                                            and max(out_top_scores) > 0) else 0.0

    return {
        "name": name,
        "Hit@1": f"{hits1}/{n}",
        "Hit@3": f"{hits3}/{n}",
        "MRR": f"{mrr_sum / n:.3f}",
        "库外拒答": f"{refused_out}/{m}",
        "库内误拒": f"{false_refusal}/{n}",
        "判定间隔": f"{margin:.2f}x",
    }


def _ranked_candidates(search, reranker, question, top_k):
    """召回（+重排）后的完整候选顺序，用于算 Hit@K —— 不受阈值过滤影响。"""
    candidates = list(search(question, top_k))
    for c in candidates:
        if c.vector_similarity is None and c.vector_distance is not None:
            c.vector_similarity = distance_to_similarity(c.vector_distance)
    if reranker is not None:
        return reranker.rerank(question, candidates, top_k).chunks
    return candidates


def main():
    print("正在加载向量库与嵌入模型……")
    from langchain_community.embeddings import DashScopeEmbeddings

    service = VectorStoreService(DashScopeEmbeddings(model=config.embedding_model_name))
    search = CachedSearch(lambda q, k: service.search_with_scores(q, k=k))
    top_k = config.retrieval_top_k

    results = [
        evaluate("A. 仅向量召回（基线）", search, None, top_k,
                 {"vector_similarity_threshold": config.vector_similarity_threshold}),
        evaluate(f"B. 向量召回 + 重排（{config.rerank_model}）", search,
                 DashScopeReranker(model=config.rerank_model), top_k,
                 {"rerank_threshold": config.rerank_threshold}),
    ]

    print()
    print("=" * 82)
    print("汇总对比")
    print("=" * 82)
    labels = [r["name"] for r in results]
    print(f"{'指标':<10}" + "".join(f"{lb:<28}" for lb in labels))
    for key in ["Hit@1", "Hit@3", "MRR", "库外拒答", "库内误拒", "判定间隔"]:
        print(f"{key:<10}" + "".join(f"{r[key]:<28}" for r in results))

    print()
    print("解读：")
    print("  · Hit@K / MRR 衡量「排得准不准」；判定间隔衡量「阈值能不能可靠工作」。")
    print("  · 判定间隔 = 库内问题的最低分 ÷ 库外问题的最高分。该值越接近 1，")
    print("    阈值就越难定，迟早要在「拒答太狠」和「什么都答」之间二选一。")
    print("  · 本仓库当前语料仅 6 个片段，Hit@K 已接近满分，重排的排序收益难以体现；")
    print("    语料扩大后需重新评测——这正是把评测脚本固化下来的意义。")
    print("=" * 82)
    return 0


if __name__ == "__main__":
    sys.exit(main())
