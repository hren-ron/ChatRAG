from typing import List

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer


class BGEEmbedder:
    """
    BGE-M3 本地 Embedding 模型。
    """

    def __init__(self, model_path):

        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model_path = str(model_path)

        self.model = SentenceTransformer(
            self.model_path,
            local_files_only=True,
            device=device
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=True
        )

        print("=" * 50)
        print("CUDA available:", torch.cuda.is_available())

        if torch.cuda.is_available():
            print("GPU:", torch.cuda.get_device_name(0))
            print("Model device:", self.model.device)
            print(
                "GPU memory allocated:",
                torch.cuda.memory_allocated() / 1024 ** 3,
                "GB"
            )

        print("=" * 50)

    def encode(self, texts: List[str], batch_size: int=32):
        return self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True
        )

    def count_token(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    @property
    def dimension(self):
        return self.model.get_embedding_dimension()
