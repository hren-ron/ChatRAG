import json
from pathlib import Path
from typing import List

from src.chunker.models import Chunk


class MetadataStore:

    def __init__(self):
        self.vector_id_to_chunk_id = []

    def build(self, chunks: List[Chunk]):

        self.vector_id_to_chunk_id = [
            chunk.chunk_id for chunk in chunks
        ]

    def get_chunk_id(self, vector_id: int):
        return self.vector_id_to_chunk_id[vector_id]

    def save(self, path: str):
        path = Path(path)

        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as f:
            json.dump(self.vector_id_to_chunk_id, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str):

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Metadata not found: {path}")

        with path.open("r", encoding="utf-8") as f:
            vector_id_to_chunk_id = json.load(f)

        store = cls()
        store.vector_id_to_chunk_id = vector_id_to_chunk_id
        return store