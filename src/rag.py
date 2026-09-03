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
from llama_index.core.llms.llm import LLM
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llmutils import LLMUtils
import asyncio
import qdrant_client

class RAGSystem(LLMUtils):
    def __init__(
        self,
        llm: LLM,
        embedding_llm: BaseEmbedding,
        context_dir: str,
        cache_dir: str,
        pipeline: List[TransformComponent]
    ):
        super().__init__(context_dir, cache_dir, pipeline, embedding_llm)
        self.llm = llm
    
    async def build(self):

        await self.run_ingestion()

        self.index = VectorStoreIndex(
            nodes=self.nodes,
            storage_context=StorageContext.from_defaults(
                vector_store=self.pipeline.vector_store,
                docstore=self.pipeline.docstore
            ),
            embed_model=self.embedding_llm
        )
        self.query_engine = self.index.as_query_engine(
            llm=self.llm,
            similarity_top_k=3
        )
    
    async def generate_response(
        self,
        query: str,
    ):
        response = await self.query_engine.aquery(query)
        answer = response.response
        source_contexts = [node.get_content() for node in response.source_nodes]
        return answer, source_contexts