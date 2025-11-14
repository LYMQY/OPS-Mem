from typing import Dict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import chromadb
from chromadb.config import Settings
from nltk.tokenize import word_tokenize
import json
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

def simple_tokenize(text):
    return word_tokenize(text)

class ChromaRetriever:
    """Vector database retrieval using ChromaDB"""
    def __init__(self, collection_name: str = "memories",model_name: str = "all-MiniLM-L6-v2"):
        """Initialize ChromaDB retriever.
        
        Args:
            collection_name: Name of the ChromaDB collection
        """
        self.client = chromadb.Client(Settings(allow_reset=True))
        self.embedding_function = SentenceTransformerEmbeddingFunction(model_name=model_name)
        self.collection = self.client.get_or_create_collection(name=collection_name,embedding_function=self.embedding_function)

    def add_cluster(self, cluster: str, metadata: Dict, cluster_id: str):
        """Add a cluster to ChromaDB with enhanced embedding using metadata.
        
        Args:
            cluster: Text content to add
            metadata: Dictionary of metadata including keywords, tags, context
            cluster_id: Unique identifier for the cluster
        """
        # Build enhanced cluster content including semantic metadata
        enhanced_cluster = cluster

        # Add problem_description information
        if 'problem_description' in metadata and metadata['problem_description'] != "General":
            enhanced_cluster += f" problem_description: {metadata['problem_description']}"

        # Add modeling_logic information
        if 'modeling_logic' in metadata and metadata['modeling_logic']:
            modeling_logic = metadata['modeling_logic'] if isinstance(metadata['modeling_logic'], list) else json.loads(metadata['modeling_logic'])
            if modeling_logic:
                enhanced_cluster += f" modeling_logic: {', '.join(modeling_logic)}"

        
        # Convert MemoryNote object to serializable format
        processed_metadata = {}
        for key, value in metadata.items():
            if isinstance(value, list):
                processed_metadata[key] = json.dumps(value)
            elif isinstance(value, dict):
                processed_metadata[key] = json.dumps(value)
            else:
                processed_metadata[key] = str(value)
        
        # Store enhanced document content for better embedding
        processed_metadata['enhanced_content'] = enhanced_cluster
                
        # Use enhanced document content for embedding generation
        self.collection.add(
            documents=[enhanced_cluster],
            metadatas=[processed_metadata],
            ids=[cluster_id]
        )
        
    def delete_document(self, cluster_id: str):
        """Delete a document from ChromaDB.
        
        Args:
            cluster_id: ID of cluster to delete
        """
        self.collection.delete(ids=[cluster_id])
        
    def search(self, query: str, k: int = 2):
        """Search for similar clusters.
        
        Args:
            query: Query text
            k: Number of results to return
            
        Returns:
            Dict with clusters, metadatas, ids, and distances
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=k
        )
        
        # Convert string metadata back to original types
        if 'metadatas' in results and results['metadatas'] and len(results['metadatas']) > 0:
            # First level is a list with one item per query
            for i in range(len(results['metadatas'])):
                # Second level is a list of metadata dicts for each result
                if isinstance(results['metadatas'][i], list):
                    for j in range(len(results['metadatas'][i])):
                        # Process each metadata dict
                        if isinstance(results['metadatas'][i][j], dict):
                            metadata = results['metadatas'][i][j]
                            for key, value in metadata.items():
                                try:
                                    # Try to parse JSON for lists and dicts
                                    if isinstance(value, str) and (value.startswith('[') or value.startswith('{')):
                                        metadata[key] = json.loads(value)
                                    # Convert numeric strings back to numbers
                                    elif isinstance(value, str) and value.replace('.', '', 1).isdigit():
                                        if '.' in value:
                                            metadata[key] = float(value)
                                        else:
                                            metadata[key] = int(value)
                                except (json.JSONDecodeError, ValueError):
                                    # If parsing fails, keep the original string
                                    pass
                        
        return results
