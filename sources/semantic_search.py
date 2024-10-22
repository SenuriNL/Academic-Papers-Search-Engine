# semantic_search.py

import numpy as np
from sentence_transformers import SentenceTransformer, util

class SemanticSearch:
    def __init__(self, papers_collection):
        # Load a pre-trained SentenceTransformer model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.papers_collection = papers_collection
        self.embeddings = self.create_embeddings(papers_collection)

    def create_embeddings(self, papers):
        # Create embeddings for the titles and abstracts of the papers
        texts = [f"{paper['Title']} {paper['Abstract']}" for paper in papers]
        embeddings = self.model.encode(texts, convert_to_tensor=True)
        return embeddings

    def search(self, query, top_k=5):
        # Encode the query
        query_embedding = self.model.encode(query, convert_to_tensor=True)

        # Calculate cosine similarities
        cos_scores = util.pytorch_cos_sim(query_embedding, self.embeddings)[0]
        
        # Get the top_k results
        top_results = np.argpartition(-cos_scores, range(top_k))[:top_k]
        
        results = []
        for idx in top_results:
            results.append((self.papers_collection[idx], cos_scores[idx].item()))
        
        # Sort results by score in descending order
        results = sorted(results, key=lambda x: x[1], reverse=True)
        return results
