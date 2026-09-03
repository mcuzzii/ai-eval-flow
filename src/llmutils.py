from pathlib import Path
from typing import List
from llama_index.core import (
    SimpleDirectoryReader, 
    VectorStoreIndex, 
    StorageContext,
    Settings
)
from llama_index.core.schema import TransformComponent
from llama_index.core.ingestion import IngestionPipeline, IngestionCache
from llama_index.core.node_parser import SentenceSplitter, MarkdownNodeParser
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.kvstore import SimpleKVStore
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.vector_stores.simple import SimpleVectorStore
import qdrant_client

class LLMUtils:
    """
    Shared logic for RAGSystem and LLMEval classes
    """

    def __init__(
        self,
        context_dir: str,
        cache_dir: str,
        pipeline: List[TransformComponent],
        embedding_llm: BaseEmbedding = None
    ):
        self.context_dir = Path(context_dir)
        self.cache_dir = Path(cache_dir)
        self.pipeline = pipeline
        self.embedding_llm = embedding_llm

        if embedding_llm is not None:
            Settings.embed_model = embedding_llm

        # Initialize persistent storage
        if not self.cache_dir.exists():
            self.cache_dir.mkdir(parents=True)
        
        # Initializes cache
        kv_path = self.cache_dir / "kv_store.json"

        if kv_path.exists():
            kv_store = SimpleKVStore.from_persist_path(kv_path)
        else:
            kv_store = SimpleKVStore()
        
        doc_path = self.cache_dir / "docstore.json"

        # Initializes docstore
        if doc_path.exists():
            docstore = SimpleDocumentStore.from_persist_path(doc_path)
        else:
            docstore = SimpleDocumentStore()
        
        cache = IngestionCache(cache=kv_store)

        vector_store_path = self.cache_dir / "vector_store.json"

        if vector_store_path.exists():
            vector_store = SimpleVectorStore.from_persist_path(vector_store_path)
        else:
            vector_store = SimpleVectorStore()

        # Initialize vector store when needed and define pipeline accordingly
        if embedding_llm is not None:

            if not isinstance(pipeline[-1], BaseEmbedding):
                pipeline.append(embedding_llm)

        self.pipeline = IngestionPipeline(
            transformations=pipeline,
            vector_store=vector_store,
            docstore=docstore,
            cache=cache
        )
    
    async def run_ingestion(self):

        # Load data
        self.documents = SimpleDirectoryReader(self.context_dir).load_data()
        
        self.nodes = await self.pipeline.arun(documents=self.documents)

        # Save the pipeline state
        self.pipeline.persist(self.cache_dir)