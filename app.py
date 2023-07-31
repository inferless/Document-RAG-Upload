import json
from sentence_transformers import SentenceTransformer, util


class InferlessPythonModel:
    def initialize(self):
        self.pipe = SentenceTransformer("sentence-transformers/paraphrase-albert-base-v2")

    def infer(self, inputs):
        sentences = inputs["sentences"]
        embeddings = self.pipe.encode(sentences)
        return {"result": embeddings}

    def finalize(self, args):
        self.pipe = None
