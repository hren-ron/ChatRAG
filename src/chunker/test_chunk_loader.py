from chunk_loader import ChunkLoader


loader = ChunkLoader(
    "../../data/output"
)

chunks = loader.load()

print(f"加载 Chunk 数量: {len(chunks)}")

for chunk in chunks[:5]:

    print("=" * 80)

    print("chunk_id:", chunk.chunk_id)
    print("document_id:", chunk.document_id)
    print("document:", chunk.document)
    print("source:", chunk.source)
    print("version:", chunk.version)

    print("chapter:", chunk.chapter)
    print("title:", chunk.title)
    print("level:", chunk.level)
    print("parent_path:", chunk.parent_path)

    print("content_hash:", chunk.content_hash)
    print("is_deleted:", chunk.is_deleted)

    print("content:")
    print(chunk.content[:300])