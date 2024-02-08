import json
import uuid
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone


class InferlessPythonModel:
    def initialize(self):
        self.pipe = SentenceTransformer("all-MiniLM-L6-v2")
        pc = Pinecone(api_key="31b47ff0-5126-4f21-9d55-8ea2714e1a7d")
        self.index = pc.Index("documents")

    def infer(self, inputs):
      data = inputs["data"]
      request_type = inputs["type"]

      if request_type=="upload":
        response = self.upload(data)
      else:
        query = inputs["data"][0]
        response = self.search(query.decode())

      return {"result":str(response)}

    def upload(self,data):
      upload_data = []
      for sentence in data:
          vectors = self.pipe.encode(sentence.decode())
          vector_data = {
                          "id": str(uuid.uuid4()),
                          "values": vectors.tolist()
                          #"metadata": {"document_category": data_row["metadata"]["category"]} #You can use store the metadata along with the vectors
                          }
          upload_data.append(vector_data)

      response = self.index.upsert(
                  vectors=upload_data,
                  namespace= "ns1")
      return response

    def search(self,query):
      vectors = self.pipe.encode(query)
      response = self.index.query(
                        namespace="ns1",         # Change the namespace according to your requirements
                        vector=vectors.tolist(), # Vectors needs to convert into a list during the search
                        top_k=2,                 # This will determine the number of documents that needs to be return
                        include_values=True,
                        include_metadata=True
                    )
      return response

    def finalize(self, args):
        self.pipe = None
        self.index = None
