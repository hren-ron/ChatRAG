from typing import List, Any


def reciprocal_rank_fusion(
    result_lists: List[List[dict[str, Any]]],
    top_k: int=10,
    k: int=60
):
    """
    使用 RRF 融合多路检索结果。

    参数：
        result_lists:
            多路已排序的检索结果。
            每条结果必须包含 chunk 和 score。
        top_k:
            最终返回的结果数量。
        k:
            RRF 平滑参数，通常从 60 开始实验。

    返回：
        按融合分数降序排列的结果。
        每条结果包含 chunk 和 RRF score。
    """


    if top_k <= 0:

        return []

    if k < 0:
        raise ValueError("k must be non-negative")

    fused_scores: dict[str, float] = {}
    chunk_map: dict[str, Any] = {}

    for results in result_lists:

        # 同一路检索中，同一个 Chunk 只计算一次
        seen_in_list: set[str] = set()

        for rank, result in enumerate(results, start=1):

            chunk = result["chunk"]
            chunk_id = chunk.chunk_id

            if chunk_id in seen_in_list:
                continue

            seen_in_list.add(chunk_id)
            chunk_map[chunk_id] = chunk

            # RRF 只看排名，不使用原始 score
            fused_scores[chunk_id] = (
                fused_scores.get(chunk_id, 0.0)
                + 1.0 / (k + rank)
            )

    ranked_ids = sorted(
        fused_scores,
        key=lambda chunk_id: fused_scores[chunk_id],
        reverse=True,
    )

    return [
        {
            "chunk": chunk_map[chunk_id],
            "score": fused_scores[chunk_id],
        }
        for chunk_id in ranked_ids[:top_k]
    ]
