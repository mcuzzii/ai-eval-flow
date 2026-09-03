from dotenv import load_dotenv
import pandas as pd
from ragas import Dataset, experiment
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory
from ragas.metrics.collections import (
    ContextPrecision,
    ContextUtilization,
    ContextRecall,
    ContextEntityRecall,
    NoiseSensitivity,
    AnswerRelevancy
)
from google import genai
from openai import AsyncOpenAI
import asyncio
import inspect
from pathlib import Path

load_dotenv()

# Limits the number of concurrent API requests
semaphore = asyncio.Semaphore(3)

# OpenRouter client
openrouter_client = AsyncOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url=os.getenv("OPENROUTER_BASE_URL"),
)

# Gemini client
google_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# LLM evaluator
judge = llm_factory(
    "openrouter/free",
    client=openrouter_client
)
embedding_generator = embedding_factory(
    'google/gemini-embedding-001',
    client=google_client
)

# Metrics are defined in the global scope to prevent re-initialization
metrics = [
    ContextPrecision(llm=judge),
    ContextUtilization(llm=judge),
    ContextRecall(llm=judge),
    ContextEntityRecall(llm=judge),
    NoiseSensitivity(llm=judge),
    AnswerRelevancy(llm=judge, embeddings=embedding_generator)
]

# Helper function to run metrics asynchronously
async def run_metric(metric, args):
    sig = inspect.signature(metric.ascore)
    target_keys = sig.parameters.keys()

    args = {k: v for k, v in args.items() if k in target_keys}

    return await metric.ascore(**args)

# Experiment configuration
@experiment()
async def run_experiment(row):
    async with semaphore:

        # Prepare arguments per row
        all_args = {
            'user_input': row['question'],
            'reference': row['ground_truth'],
            'response': row['answer'],
            'retrieved_contexts': [row['context']]
        }

        # Create and run concurrent metrics
        tasks = [run_metric(metric, all_args) for metric in metrics]
        results = await asyncio.gather(*tasks)

        # Store and return results
        scores_dict = {m.name + '_score': r.value for m, r in zip(metrics, results)}
        experiment_view = {
            **row,
            **scores_dict
        }

        return experiment_view

async def main():

    # Create dataset from markdown file
    read_markdown(
        file_path='evals/datasets/qa_data.md',
        output_path='evals/datasets/qa_data.csv',
        ignore_cache=True
    )

    # Load dataset
    qa_data = Dataset.load(
        name="qa_data",
        backend="local/csv",
        root_dir="evals",
    )

    # Run the experiment
    result = await run_experiment.arun(qa_data)

    # Save results
    result_df = result.to_pandas()
    result_df.to_csv('evals/experiments/experiment_results.csv')

if __name__ == "__main__":
    asyncio.run(main())