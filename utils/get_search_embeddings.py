# utils/embedding_utils.py
import torch
from sentence_transformers import SentenceTransformer
import numpy as np


class EmbeddingGenerator:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name).to(device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def generate_embedding(self, text: str) -> np.ndarray:
        return self.model.encode(text)

    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
