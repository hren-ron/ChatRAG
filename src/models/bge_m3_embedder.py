from typing import List

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer


class BGEEmbedder:
    """
    BGE-M3 本地 Embedding 模型。
    """

    def __init__(self, model_path):

        self.model_path = str(model_path)

        self.model = SentenceTransformer(self.model_path, local_files_only=True)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=True
        )

    def encode(self, texts: List[str], batch_size: int=32):
        return self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True)

    def count_token(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    @property
    def dimension(self):
        return self.model.get_embedding_dimension()
