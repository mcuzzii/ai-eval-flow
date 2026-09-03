import os
from google import genai
from ragas.llms import llm_factory
from ragas.embeddings import GoogleEmbeddings
from ragas.metrics.collections import AnswerCorrectness, ContextPrecision
import asyncio
from dotenv import load_dotenv

load_dotenv()

# Create client (new SDK)
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# Create LLM
llm = llm_factory("gemini-2.5-flash-live", client=client, provider='google')

# Create embeddings using the same client
embeddings = GoogleEmbeddings(client=client, model="gemini-embedding-001")

# Create metrics with explicit LLM and embeddings
metrics = [
    ContextPrecision(llm=llm),  # LLM-only metric
    AnswerCorrectness(llm=llm, embeddings=embeddings),  # Needs both
]

# Use metrics with your evaluation workflow

async def main():
    result = await metrics[1].ascore(
        user_input="What is the capital of France?",
        response="Paris",
        reference="Paris is the capital of France."
    )
    print(result)

if __name__ == "__main__":
    # asyncio.run(main())
    for model in client.models.list():
        print(model.name)