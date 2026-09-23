from __future__ import annotations

from langchain_chroma import Chroma

import config_data as config
from retrieval import RetrievedChunk, distance_to_similarity


class VectorStoreService(object):
    def __init__(self, embedding):
        """
        :param embedding: 嵌入模型的传入
        """
        self.embedding = embedding

        self.vector_store = Chroma(
            collection_name=config.collection_name,
            embedding_function=self.embedding,
            persist_directory=config.persist_directory,
        )

    def search_with_scores(self, query: str, k: int = None) -> list:
        """带分数的向量召回，返回统一结构的 `RetrievedChunk`。

        原实现用 `as_retriever()` 只拿文档正文、丢掉了距离分数，导致下游
        既无法做阈值门控、也无从判断召回质量。这里改用
        `similarity_search_with_score()`，把距离一并带出来并归一化为相似度。
        """
        k = int(k or config.retrieval_top_k)
        pairs = self.vector_store.similarity_search_with_score(query, k=k)

        chunks = []
        for doc, distance in pairs:
            d = float(distance)
            chunks.append(
                RetrievedChunk(
                    content=doc.page_content,
                    metadata=dict(doc.metadata or {}),
                    vector_distance=d,
                    vector_similarity=distance_to_similarity(d),
                )
            )
        # Chroma 按距离升序返回，等价于相似度降序；这里显式排序，避免依赖底层实现细节
        chunks.sort(key=lambda c: c.score, reverse=True)
        return chunks

    def get_retriever(self):
        """[保留] 兼容旧调用：不带分数、不做门控的裸检索器。

        新链路请优先使用 `search_with_scores()` + `RetrievalPipeline`。
        """
        return self.vector_store.as_retriever(search_kwargs={"k": config.retrieval_top_k})


if __name__ == '__main__':
    from langchain_community.embeddings import DashScopeEmbeddings

    service = VectorStoreService(DashScopeEmbeddings(model=config.embedding_model_name))
    for chunk in service.search_with_scores("我的身高180，尺码推荐"):
        print(f"[{chunk.score:.4f}] {chunk.preview(50)}")
