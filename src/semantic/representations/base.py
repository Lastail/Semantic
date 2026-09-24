from abc import ABC, abstractmethod
class RepresentationExtractor(ABC):
    @abstractmethod
    def encode(self, texts, max_tokens): ...
