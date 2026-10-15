from abc import ABC, abstractmethod

class BaseRepresentation(ABC):
    """For fixed (T,L), return TWO-NN first/second neighbour distances."""
    @abstractmethod
    def compute_neighbors(self, texts, length, cache_path=None):
        raise NotImplementedError
