from langchain.embeddings import VertexAIEmbeddings
from typing import List
import time 

class CustomVertexAIEmbeddings(VertexAIEmbeddings):
    requests_per_minute: int
    num_instances_per_batch: int
    
    # Utility functions for Embeddings API with rate limiting
    def rate_limit(self,max_per_minute):
        period = 60 / max_per_minute
        print("Waiting")
        while True:
            before = time.time()
            yield
            after = time.time()
            elapsed = after - before
            sleep_time = max(0, period - elapsed)
            if sleep_time > 0:
                print(".", end="")
                time.sleep(sleep_time)
                
    # Overriding embed_documents method
    def embed_documents(self, texts: List[str]):
        limiter = self.rate_limit(self.requests_per_minute)
        results = []
        docs = list(texts)

        while docs:
            # Working in batches because the API accepts maximum 5
            # documents per request to get embeddings
            head, docs = (
                docs[: self.num_instances_per_batch],
                docs[self.num_instances_per_batch :],
            )
            chunk = self.client.get_embeddings(head)
            results.extend(chunk)
            next(limiter)

        return [r.values for r in results]