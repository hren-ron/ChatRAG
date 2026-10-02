from transformers import AutoTokenizer

from src.chunker.chunk_loader import ChunkLoader
from src.chunker.secondary_chunker import SecondaryChunker


# ============================================================
# 配置
# ============================================================

# BGE-M3
MODEL_NAME = "BAAI/bge-m3"

# Structured Chunk 输入目录
INPUT_DIR = ("../../data/txt")

# Secondary Chunker 参数
MAX_TOKENS = 600
MIN_TOKENS = 100
OVERLAP_TOKENS = 80

# BGE-M3 最大输入长度
MODEL_MAX_TOKENS = 8192


# ============================================================
# Tokenizer
# ============================================================

print("=" * 80)
print("Loading tokenizer...")
print("=" * 80)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
)

print("Tokenizer loaded.")
print()


# ============================================================
# ChunkLoader
# ============================================================

print("=" * 80)
print("Loading Structured Chunks...")
print("=" * 80)

loader = ChunkLoader(
    input_dir=INPUT_DIR
)

structured_chunks = loader.load()

print(f"Structured Chunks: {len(structured_chunks)}")
print()


# ============================================================
# SecondaryChunker
# ============================================================

chunker = SecondaryChunker(
    tokenizer=tokenizer,
    max_tokens=MAX_TOKENS,
    min_tokens=MIN_TOKENS,
    overlap_tokens=OVERLAP_TOKENS,
)


# ============================================================
# 统计变量
# ============================================================

total_retrieval_chunks = 0

structured_token_counts = []
retrieval_token_counts = []

too_long_structured = []
too_long_retrieval = []
over_max_tokens = []

empty_retrieval_chunks = []

# 保存部分样例
samples = []


# ============================================================
# 开始二次切分
# ============================================================

print("=" * 80)
print("Running SecondaryChunker...")
print("=" * 80)

for index, chunk in enumerate(structured_chunks, start=1):

    # --------------------------------------------------------
    # 统计 Structured Chunk Token
    # --------------------------------------------------------

    structured_token_ids = tokenizer.encode(
        chunk.content,
        add_special_tokens=False,
    )

    structured_token_count = len(structured_token_ids)

    structured_token_counts.append(
        structured_token_count
    )

    # Structured Chunk 超过模型限制
    if structured_token_count > MODEL_MAX_TOKENS:
        too_long_structured.append(
            {
                "chunk_id": chunk.chunk_id,
                "token_count": structured_token_count,
            }
        )

    # --------------------------------------------------------
    # Secondary Chunking
    # --------------------------------------------------------

    retrieval_chunks = chunker.split(chunk)

    total_retrieval_chunks += len(retrieval_chunks)

    # --------------------------------------------------------
    # 检查 Retrieval Chunk
    # --------------------------------------------------------

    for retrieval_chunk in retrieval_chunks:

        token_count = retrieval_chunk.token_count

        # 如果模型没有设置 token_count，重新计算
        if token_count is None:
            token_count = len(
                tokenizer.encode(
                    retrieval_chunk.content,
                    add_special_tokens=False,
                )
            )

        retrieval_token_counts.append(
            token_count
        )

        # 空 Chunk
        if not retrieval_chunk.content.strip():
            empty_retrieval_chunks.append(
                retrieval_chunk.chunk_id
            )

        # 超过 SecondaryChunker 最大长度
        if token_count > MAX_TOKENS:
            over_max_tokens.append(
                {
                    "chunk_id": retrieval_chunk.chunk_id,
                    "token_count": token_count,
                }
            )

        # 超过 BGE-M3 最大输入长度
        if token_count > MODEL_MAX_TOKENS:
            too_long_retrieval.append(
                {
                    "chunk_id": retrieval_chunk.chunk_id,
                    "token_count": token_count,
                }
            )

    # --------------------------------------------------------
    # 保存前几个样例
    # --------------------------------------------------------

    if len(samples) < 5 and retrieval_chunks:

        samples.append(
            {
                "source_chunk": chunk,
                "retrieval_chunks": retrieval_chunks,
            }
        )

    # --------------------------------------------------------
    # 进度
    # --------------------------------------------------------

    if index % 100 == 0:
        print(
            f"Processed {index}/{len(structured_chunks)} "
            f"Structured Chunks"
        )


# ============================================================
# 统计函数
# ============================================================

def print_statistics(title, values):

    if not values:
        print(f"{title}: 0")
        return

    total = len(values)
    minimum = min(values)
    maximum = max(values)
    average = sum(values) / total

    print(f"{title}")
    print(f"  Count   : {total}")
    print(f"  Min     : {minimum}")
    print(f"  Max     : {maximum}")
    print(f"  Average : {average:.2f}")


# ============================================================
# 输出统计结果
# ============================================================

print()
print("=" * 80)
print("RESULT")
print("=" * 80)

print()

print(f"Structured Chunks : {len(structured_chunks)}")
print(f"Retrieval Chunks  : {total_retrieval_chunks}")

print()

print_statistics(
    "Structured Chunk Token Statistics:",
    structured_token_counts,
)

print()

print_statistics(
    "Retrieval Chunk Token Statistics:",
    retrieval_token_counts,
)

print()

print("=" * 80)
print("Validation")
print("=" * 80)

print()

print(
    f"Structured Chunks > {MODEL_MAX_TOKENS}: "
    f"{len(too_long_structured)}"
)

print(
    f"Retrieval Chunks > {MAX_TOKENS}: "
    f"{len(over_max_tokens)}"
)

print(
    f"Retrieval Chunks > {MODEL_MAX_TOKENS}: "
    f"{len(too_long_retrieval)}"
)

print(
    f"Empty Retrieval Chunks: "
    f"{len(empty_retrieval_chunks)}"
)

print()


# ============================================================
# 输出超长 Structured Chunk
# ============================================================

if too_long_structured:

    print("=" * 80)
    print(
        f"Structured Chunks > {MODEL_MAX_TOKENS}"
    )
    print("=" * 80)

    for item in too_long_structured[:20]:
        print(
            f"{item['chunk_id']}: "
            f"{item['token_count']} tokens"
        )

    if len(too_long_structured) > 20:
        print(
            f"... and "
            f"{len(too_long_structured) - 20} more"
        )

    print()


# ============================================================
# 输出超过 SecondaryChunker 限制的 Retrieval Chunk
# ============================================================

if over_max_tokens:

    print("=" * 80)
    print(
        f"Retrieval Chunks > {MAX_TOKENS}"
    )
    print("=" * 80)

    for item in over_max_tokens[:20]:
        print(
            f"{item['chunk_id']}: "
            f"{item['token_count']} tokens"
        )

    if len(over_max_tokens) > 20:
        print(
            f"... and "
            f"{len(over_max_tokens) - 20} more"
        )

    print()


# ============================================================
# 输出超过 BGE-M3 最大长度的 Retrieval Chunk
# ============================================================

if too_long_retrieval:

    print("=" * 80)
    print(
        f"ERROR: Retrieval Chunks > {MODEL_MAX_TOKENS}"
    )
    print("=" * 80)

    for item in too_long_retrieval[:20]:
        print(
            f"{item['chunk_id']}: "
            f"{item['token_count']} tokens"
        )

    if len(too_long_retrieval) > 20:
        print(
            f"... and "
            f"{len(too_long_retrieval) - 20} more"
        )

    print()


# ============================================================
# 输出样例
# ============================================================

print("=" * 80)
print("SAMPLES")
print("=" * 80)

for sample_index, sample in enumerate(samples, start=1):

    source_chunk = sample["source_chunk"]
    retrieval_chunks = sample["retrieval_chunks"]

    print()
    print("-" * 80)

    print(
        f"Sample {sample_index}"
    )

    print(
        f"Source Chunk ID: "
        f"{source_chunk.chunk_id}"
    )

    print(
        f"Source Tokens: "
        f"{len(tokenizer.encode(source_chunk.content, add_special_tokens=False))}"
    )

    print(
        f"Retrieval Chunks: "
        f"{len(retrieval_chunks)}"
    )

    print()

    for retrieval_chunk in retrieval_chunks[:3]:

        print(
            f"[{retrieval_chunk.chunk_id}] "
            f"{retrieval_chunk.token_count} tokens"
        )

        print(
            f"Title: "
            f"{retrieval_chunk.title}"
        )

        print(
            f"Parent Path: "
            f"{retrieval_chunk.parent_path}"
        )

        print(
            "Content Preview:"
        )

        preview = retrieval_chunk.content[:300]

        print(preview)

        if len(retrieval_chunk.content) > 300:
            print("...")

        print()


# ============================================================
# 最终结果
# ============================================================

print("=" * 80)
print("FINAL")
print("=" * 80)

if too_long_retrieval:
    print(
        "FAILED: Some Retrieval Chunks exceed "
        f"{MODEL_MAX_TOKENS} tokens."
    )

elif over_max_tokens:
    print(
        "WARNING: Some Retrieval Chunks exceed "
        f"SecondaryChunker max_tokens={MAX_TOKENS}."
    )

else:
    print(
        "PASSED: All Retrieval Chunks are within "
        f"{MAX_TOKENS} tokens."
    )

print()

print(
    "Next step: "
    "Embedding can be performed on Retrieval Chunks."
)