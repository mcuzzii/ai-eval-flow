"""
This module generates structured question-answer (Q&A) pairs from a given context.
"""

from pydantic import BaseModel, Field
from typing import List
from functools import wraps
from ragas.prompt import PydanticPrompt
from ragas.metrics.collections import AnswerRelevancy
from llama_index.core.schema import TransformComponent
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
import pandas as pd
from pathlib import Path
import json
import asyncio
import inspect
import functools
import joblib
import re
from ragas.llms.base import BaseRagasLLM
from ragas.embeddings.base import BaseRagasEmbeddings
from ragas.metrics.base import Metric
from llmutils import LLMUtils
from rag import RAGSystem
from ragas import Dataset, experiment

# Output data format
class QAPair(BaseModel):
    question: str = Field(
        description="A question based on the context"
    )
    answer: str = Field(
        description="The answer derived from the context"
    )
    answer_type: str = Field(
        description="The kind of answer derived as appropriate to the question, which can only be factual, definitional, or reasoning"
    )

# Input data format, with fields to guardrail against markdown language
class ContextInput(BaseModel):
    text_chunk: str = Field(
        description="This is a chunk of Markdown text. Ignore the formatting and focus on the technical details for your questions."
    )
    num_questions: int = Field(
        description="The specific number of Q&A pairs to generate"
    )

# For containing QAPair objects as a single dictionary item
class QASet(BaseModel):
    items: List[QAPair]

# Q&A prompt template for Q&A generation with variable number of Q&A pairs per chunk
class VariableQAPrompt(PydanticPrompt[ContextInput, QASet]):
    instruction = """
    Read the following markdown text chunk:
    {text_chunk}

    Based on this content, generate exactly {number_of_questions} question and answer pairs.
    Ensure the questions are specific and the answers are derived solely from the text.
    """
    input_model = ContextInput
    output_model = QASet

class LLMEval(LLMUtils):

    def __init__(
        self,
        test_llm: RAGSystem,
        judge_llm: BaseRagasLLM,
        embedding_llm: BaseRagasEmbeddings,
        context_dir: str,
        cache_dir: str,
        pipeline: List[TransformComponent],
        metrics: List[Metric]
    ):
        super().__init__(context_dir, cache_dir, pipeline)
        self.test_llm = test_llm
        self.judge_llm = judge_llm
        self.embedding_llm = embedding_llm
        self.data = []
        self._prompt = VariableQAPrompt()
        self._history = []

        args = {
            'llm': self.judge_llm,
            'embeddings': self.embedding_llm,
        }

        self.metrics = []

        for metric in metrics:
            init_kwargs = {'llm': self.judge_llm}

            if issubclass(metric, AnswerRelevancy):
                init_kwargs['embeddings'] = self.embedding_llm
            
            try:
                self.metrics.append(metric(**init_kwargs))
            except TypeError as e:
                print(f"Could not init {metric.__name__}: {e}")
    
    async def _generate_qa(
        self,
        input_data: ContextInput
    ):
        # Generate the structured response
        qa_pairs = []
        try:
            result = await self._prompt.generate(llm=self.judge_llm, data=input_data)
            for item in result.items:
                qa_dict = item.model_dump()
                qa_dict['answer_reference'] = input_data.text_chunk
                qa_pairs.append(qa_dict)
            return qa_pairs
        
        except Exception as e:
            print(f"Error processing chunk: {e}")
            return
    
    def _cache_method(method):
        @wraps(method)
        async def wrapper(self, *args, **kwargs):
            cache_file = self.cache_dir / 'qa.json'
            cache_history = self.cache_dir / 'qa_history.joblib'

            if cache_file.exists() and cache_history.exists():
                with open(cache_file, 'r', encoding='utf-8') as f:
                    self.data = json.load(f)
                self._history = joblib.load(cache_history)
                if method.__qualname__ in self._history:
                    return
            
            await method(self, *args, **kwargs)

            self._history.append(method.__qualname__)
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=4)
            joblib.dump(self._history, cache_history)
        return wrapper
    
    @_cache_method
    async def generate_examination_set(self):

        final_dataset = []
        
        tasks = []
        for chunk in self.nodes:
            # Business logic: 1 question per 400 characters (min 1, max 5)
            num_to_gen = max(1, min(len(chunk.text) // 400, 5))
            
            # Prepare the input data object
            input_data = ContextInput(
                text_chunk=chunk.text,
                num_questions=num_to_gen
            )

            # Generate the structured response
            tasks.append(self._generate_qa(input_data))
        
        results = await asyncio.gather(*tasks)

        for result in results:
            if result is not None:
                final_dataset.extend(result)

        self.data = final_dataset
    
    @_cache_method
    async def generate_test_responses(self):
        final_dataset = []

        tasks = []

        await self.test_llm.build()

        for qa_pair in self.data:
            task = self.test_llm.generate_response(qa_pair['question'])
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)

        for i, (answer, source_contexts) in enumerate(results):
            self.data[i]['response'] = answer
            self.data[i]['context'] = source_contexts
    
    async def _run_metric(self,metric, args):
        sig = inspect.signature(metric.ascore)
        target_keys = sig.parameters.keys()

        args = {k: v for k, v in args.items() if k in target_keys}

        return await metric.ascore(**args)

    # Experiment configuration
    async def _run_experiment(self, row):

        # Prepare arguments per row
        all_args = {
            'user_input': row['question'],
            'reference': row['answer'],
            'response': row['response'],
            'retrieved_contexts': row['context']
        }

        # Create and run concurrent metrics
        tasks = [self._run_metric(metric, all_args) for metric in self.metrics]
        results = await asyncio.gather(*tasks)

        # Store and return results
        scores_dict = {m.name + '_score': r.value for m, r in zip(self.metrics, results)}
        experiment_view = {
            **row,
            **scores_dict
        }

        return experiment_view
    
    @_cache_method
    async def run_evaluation(self):

        tasks = []

        for row in self.data:
            tasks.append(self._run_experiment(row=row))
        
        self.data = await asyncio.gather(*tasks)
    
    