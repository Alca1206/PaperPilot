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





class FakeChatClient:
    """Stands in for ollama.Client.chat: records calls, returns queued replies in order."""

    def __init__(self, replies=("OK",), error=None):
        self.calls = []
        self._replies = list(replies)
        self._error = error

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        content = self._replies.pop(0) if self._replies else ""
        return SimpleNamespace(message=SimpleNamespace(content=content))

class FakeChatClient:
    """Stands in for ollama.Client.chat: records calls, returns queued replies in order."""

    def __init__(self, replies=("OK",), error=None):
        self.calls = []
        self._replies = list(replies)
        self._error = error

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        content = self._replies.pop(0) if self._replies else ""
        return SimpleNamespace(message=SimpleNamespace(content=content))




class KeywordEmbedClient:
    """A tiny "meaning" model for tests: one vector dimension per keyword.

    Texts that share keywords get similar vectors, so search results are
    predictable. The final constant dimension avoids all-zero vectors.
    """

    def __init__(self, vocabulary):
        self.vocabulary = [word.lower() for word in vocabulary]
        self.calls = []

    def embed(self, model, input):
        self.calls.append({"model": model, "input": list(input)})
        vectors = []
        for text in input:
            lowered = text.lower()
            vectors.append([float(lowered.count(word)) for word in self.vocabulary] + [0.1])
        return SimpleNamespace(embeddings=vectors)