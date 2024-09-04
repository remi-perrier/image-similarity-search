from abc import ABC, abstractmethod

import numpy as np


class VectorDatabase(ABC):
    """Abstract Base Class for a vector database."""

    vector_size: int

    @abstractmethod
    def load(self, filepath: str) -> None:
        """Load the database from a file."""
        pass

    @abstractmethod
    def save(self, filepath: str) -> None:
        """Save the database to a file."""
        pass

    @abstractmethod
    def add(self, image_vector: np.ndarray, id: int, metadata: dict) -> None:
        """Add a new vector with associated ID and metadata."""
        pass

    @abstractmethod
    def delete(self, id: int) -> None:
        """Delete a vector and its associated data by ID."""
        pass

    @abstractmethod
    def top_k_similarities(self, image_vector: np.ndarray, k: int) -> list[int]:
        """Find the top K most similar vectors."""
        pass


class NumpyVectorDatabase(VectorDatabase):
    """A concrete implementation of VectorDatabase using NumPy."""

    def __init__(self, vector_size: int) -> None:
        super().__init__()
        self.vector_size = vector_size
        self.ids = np.empty((0,), dtype=int)
        self.vectors = np.empty((0, vector_size))
        self.metadata: list[dict] = []

    def load(self, filepath: str) -> None:
        """Load the database from a file."""
        data = np.load(filepath, allow_pickle=True)
        self.ids = data["ids"]
        self.vectors = data["vectors"]
        self.metadata = data["metadata"].tolist()

    def save(self, filepath: str) -> None:
        """Save the database to a file."""
        np.savez(
            filepath,
            ids=self.ids,
            vectors=self.vectors,
            metadata=np.array(self.metadata, dtype=object),
        )

    def add(self, image_vector: np.ndarray, id: int, metadata: dict) -> None:
        """Add a new vector with associated ID and metadata."""
        assert image_vector.shape == (self.vector_size,), "Vector dimension mismatch."
        self.ids = np.append(self.ids, id)
        self.vectors = np.vstack([self.vectors, image_vector])
        self.metadata.append(metadata)

    def delete(self, id: int) -> None:
        """Delete a vector and its associated data by ID."""
        idx = np.where(self.ids == id)[0]
        if idx.size == 0:
            raise ValueError(f"ID {id} not found in database.")
        # Get the actual index value.
        idx = idx[0]
        self.ids = np.delete(self.ids, idx)
        self.vectors = np.delete(self.vectors, idx, axis=0)
        del self.metadata[idx]

    def top_k_similarities(self, image_vector: np.ndarray, k: int = 3) -> list[int]:
        """Find the top K most similar vectors."""
        assert image_vector.shape == (
            self.vector_size,
        ), "Query vector dimension mismatch."
        distances = self.cosine_similarity(image_vector, self.vectors)
        # Get indices of top K similar vectors
        top_k_indices = np.argsort(distances)[-k:][::-1]
        return self.ids[top_k_indices].tolist()

    @staticmethod
    def cosine_similarity(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between the query and all vectors in the database."""
        dot_product = np.dot(vectors, query)
        norm_query = np.linalg.norm(query)
        norm_vectors = np.linalg.norm(vectors, axis=1)
        return dot_product / (norm_query * norm_vectors)
