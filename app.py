import os
from langchain.document_loaders import OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import Pinecone
import nltk


class InferlessPythonModel:
    def initialize(self):
        nltk.download('punkt_tab')
        nltk.download('averaged_perceptron_tagger_eng')
        #define the index name of Pinecone, embedding model name and pinecone API KEY
        index_name = "documents"
        embed_model_id = "sentence-transformers/all-MiniLM-L6-v2"
        os.environ["PINECONE_API_KEY"] = "YOUR_PINECONE_API_KEY"

        #Initialize the embedding model, text_splitter & pinecone
        embeddings=HuggingFaceEmbeddings(model_name=embed_model_id)
        self.text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        self.pinecone = Pinecone(index_name=index_name, embedding=embeddings)

    def infer(self, inputs):
      pdf_link = inputs["pdf_url"]
      loader = OnlinePDFLoader(pdf_link)
      data = loader.load()
      documents = self.text_splitter.split_documents(data)
      response = self.pinecone.add_documents(documents)
      
      return {"result":response}

    def finalize(self):
      pass
