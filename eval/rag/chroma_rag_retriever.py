"""
Chroma-backed RAG retriever for CBT knowledge base.

This module provides a simple wrapper to index the existing JSON knowledge
base into a Chroma collection using sentence-transformers embeddings and
query it for semantic nearest neighbors.

Note: This requires `chromadb` and `sentence-transformers` to be installed.
"""
from pathlib import Path
from typing import List, Dict, Optional

try:
    import chromadb
    from chromadb.config import Settings
except Exception as e:  # pragma: no cover - informative at runtime
    chromadb = None

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:  # pragma: no cover - informative at runtime
    SentenceTransformer = None

import json
import os


class ChromaCBTRetriever:
    def __init__(
        self,
        knowledge_base_dir: str,
        persist_directory: Optional[str] = None,
        embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ):
        if chromadb is None or SentenceTransformer is None:
            raise RuntimeError(
                "Missing dependencies: install chromadb and sentence-transformers to use Chroma retriever"
            )

        self.kb_dir = Path(knowledge_base_dir)
        self.persist_directory = persist_directory
        self.embedding_model_name = embedding_model_name
        self.model = SentenceTransformer(embedding_model_name)

        # Initialize client (if persist_directory provided, use that path)
        if persist_directory:
            settings = Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist_directory)
            self.client = chromadb.Client(settings)
        else:
            self.client = chromadb.Client()

        # Single collection name for CBT combined KB
        self.collection_name = "cbt_kb_collection"
        # Create or get collection
        try:
            self.collection = self.client.get_collection(self.collection_name)
        except Exception:
            self.collection = self.client.create_collection(self.collection_name)

    def build_collection(self, overwrite: bool = False) -> None:
        """Index cognitive frameworks and intervention strategies into Chroma."""
        if overwrite:
            try:
                self.client.delete_collection(self.collection_name)
            except Exception:
                pass
            self.collection = self.client.create_collection(self.collection_name)

        # Load sources
        frameworks_file = self.kb_dir / "cognitive_frameworks.json"
        strategies_file = self.kb_dir / "intervention_strategies.json"

        docs: List[str] = []
        metadatas: List[Dict] = []
        ids: List[str] = []

        # index frameworks
        if frameworks_file.exists():
            with open(frameworks_file, 'r', encoding='utf-8') as f:
                frameworks = json.load(f)
            for i, fw in enumerate(frameworks):
                text = fw.get('event', '') + '\n' + '\n'.join(fw.get('automatic_thoughts', []))
                meta = {**fw, 'kb_type': 'framework', 'kb_index': i}
                docs.append(text)
                metadatas.append(meta)
                ids.append(f'framework_{i}')

        # index strategies
        if strategies_file.exists():
            with open(strategies_file), 'r', encoding='utf-8') as f:  # type: ignore
                pass
        # safer open separately to avoid lint issues
        if strategies_file.exists():
            with open(strategies_file, 'r', encoding='utf-8') as f:
                strategies = json.load(f)
            for i, s in enumerate(strategies):
                text = (s.get('theme', '') or '') + '\n' + '\n'.join(s.get('case_material', [])[:3])
                meta = {**s, 'kb_type': 'strategy', 'kb_index': i}
                docs.append(text)
                metadatas.append(meta)
                ids.append(f'strategy_{i}')

        if not docs:
            return

        # compute embeddings
        embeddings = self.model.encode(docs, show_progress_bar=True)

        # add to collection
        # collection.add may accept embeddings param depending on chromadb version
        try:
            self.collection.add(ids=ids, documents=docs, metadatas=metadatas, embeddings=embeddings)
        except TypeError:
            # older/newer chromadb API differences
            self.collection.add(ids=ids, documents=docs, metadatas=metadatas)

    def query(self, question: str, top_k: int = 3) -> Dict:
        """Query the Chroma collection and return nearest items with metadata."""
        emb = self.model.encode([question])[0]

        # Query collection
        try:
            res = self.collection.query(query_embeddings=[emb], n_results=top_k, include=['metadatas', 'documents', 'distances'])
            # result format may vary; normalize
            metadatas = res.get('metadatas', [[]])[0]
            documents = res.get('documents', [[]])[0]
            distances = res.get('distances', [[]])[0]
        except Exception:
            # fallback to different key names
            res = self.collection.query(query_embeddings=[emb], n_results=top_k)
            metadatas = res['metadatas'][0] if 'metadatas' in res else []
            documents = res['documents'][0] if 'documents' in res else []
            distances = res.get('distances', [[]])[0]

        items = []
        for doc, meta, dist in zip(documents, metadatas, distances):
            items.append({'text': doc, 'metadata': meta, 'distance': dist})

        return {'query': question, 'results': items}
