from typing import Optional

from pydantic import Field, BaseModel


class Chunk(BaseModel):

    """
    Chapter Splitter 输出的基础 Chunk。

    ChunkLoader 只负责：
        TXT -> Structured Chunk

    不负责：
        - 二次切分
        - Token 计算
        - Embedding
        - Vector DB
    """

    # =========================
    # Chunk 身份
    # =========================

    chunk_id: str = Field(min_length=1)

    # =========================
    # 文档信息
    # =========================

    document_id: str = Field(min_length=1)
    document: Optional[str] =  None
    source: Optional[str] =  None
    version: Optional[str] =  None

    # =========================
    # 章节信息
    # =========================

    chapter: Optional[str] =  None
    title: Optional[str] =  None
    level: Optional[int] = Field(default=None, ge=0, le=4)
    parent_path: Optional[str] = None

    # =========================
    # 内容
    # =========================

    content: str = Field(min_length=1)
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    # =========================
    # 状态
    # =========================

    is_deleted: bool = False

    # SecondaryChunker
    chunk_strategy: Optional[str] = None
    chunk_index: Optional[int] = Field(default=None, ge=1)
    token_count: Optional[int] = Field(default=None, ge=1)

    # Embedding
    embedding_model: Optional[str] = None
    embedding_model_version: Optional[str] = None
    embedding_dimension: Optional[int] = Field(default=None, gt=0)

    # =========================
    # 页码信息
    # =========================

    page_start: Optional[int] = Field(default=None, ge=1)

    page_end: Optional[int] = Field(default=None, ge=1)