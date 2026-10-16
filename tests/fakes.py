"""Fake stand-ins for external services, shared by several test files."""

from types import SimpleNamespace


class FakeEmbedClient:
    """Stands in for ollama.Client.embed: records calls, returns one small vector per text."""

    def __init__(self, error=None, drop_one=False):
        self.calls = []
        self._error = error
        self._drop_one = drop_one

    def embed(self, model, input):
        self.calls.append({"model": model, "input": list(input)})
        if self._error is not None:
            raise self._error
        vectors = [[float(len(text)), 1.0, 0.0] for text in input]
        if self._drop_one:
            vectors = vectors[:-1]
        return SimpleNamespace(embeddings=vectors)