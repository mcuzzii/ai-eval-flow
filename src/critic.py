import asyncio
import json
import ollama
from datetime import datetime

from state import AppState, STATUS_RUNNING, STATUS_FLAGGED, STATUS_APPROVED, STATUS_PENDING

CRITIC_MODEL = "qwen3.6:35b"
THINK        = True

CRITIQUE_PROMPT = """\
You are a rigorous QA evaluator for LLM training data.

Given a question, an answer, and the reference text chunk the answer was \
based on, critically evaluate the answer.

Return ONLY valid JSON — no markdown fences, no preamble, no extra text:
{{
  "accuracy":      <1-5>,
  "completeness":  <1-5>,
  "clarity":       <1-5>,
  "issues":        ["issue 1", "issue 2"],
  "needs_edit":    true/false,
  "summary":       "<one sentence: main problem, or 'No issues found'>"
}}

Rubric:
- accuracy (1-5):     Are all claims faithful to the reference? 5=perfect, 1=hallucinations.
- completeness (1-5): Are all key reference points covered? 5=comprehensive, 1=major gaps.
- clarity (1-5):      Is the answer clear and appropriately concise? 5=excellent, 1=confusing.
- needs_edit:         true if ANY score < 4 or any issues found.

Question:  {question}
Answer:    {answer}
Reference: {reference}"""


def _strip_thinking(text: str) -> str:
    """Remove <think>…</think> blocks that Qwen emits in thinking mode."""
    if "<think>" in text and "</think>" in text:
        end = text.rfind("</think>")
        text = text[end + 8:].strip()
    return text


def _parse_critique(raw: str) -> dict:
    """Strip fences and parse JSON, raising on failure."""
    raw = raw.strip()
    # strip ```json or ``` wrappers
    for prefix in ("```json", "```"):
        if raw.startswith(prefix):
            raw = raw[len(prefix):]
    if raw.endswith("```"):
        raw = raw[:-3]
    return json.loads(raw.strip())


async def _critique_one(
    pair,
    client: ollama.AsyncClient,
    semaphore: asyncio.Semaphore,
    retries: int = 2,
):
    prompt = CRITIQUE_PROMPT.format(
        question  = pair.question,
        answer    = pair.answer,
        reference = pair.reference,
    )
    options = {"think": True} if THINK else {}

    for attempt in range(retries + 1):
        pair.status = STATUS_RUNNING
        try:
            async with semaphore:
                response = await client.chat(
                    model    = CRITIC_MODEL,
                    messages = [{"role": "user", "content": prompt}],
                    options  = options,
                )

            raw = _strip_thinking(response.message.content)
            critique = _parse_critique(raw)

            pair.critique = critique
            pair.status   = STATUS_FLAGGED if critique.get("needs_edit") else STATUS_APPROVED
            if pair.status == STATUS_FLAGGED:
                from datetime import datetime
                pair.flagged_at = datetime.now().isoformat()
            return

        except json.JSONDecodeError:
            if attempt == retries:
                pair.status = STATUS_PENDING   # give up, keep retryable
        except Exception:
            if attempt == retries:
                pair.status = STATUS_PENDING
            else:
                await asyncio.sleep(2 ** attempt)   # brief back-off


async def _worker(worker_id: int, queue: asyncio.Queue,
                  client: ollama.AsyncClient, semaphore: asyncio.Semaphore):
    while True:
        try:
            pair = queue.get_nowait()
        except asyncio.QueueEmpty:
            break
        await _critique_one(pair, client, semaphore)
        queue.task_done()


async def run_critics(app: AppState):
    """
    Entry point called from the background thread.
    Enqueues all pending pairs and runs MAX_CONCURRENT workers.
    """
    queue  = asyncio.Queue()
    client = ollama.AsyncClient()

    for pair in app.pairs:
        if pair.status == STATUS_PENDING:
            await queue.put(pair)

    semaphore = asyncio.Semaphore(app.max_concurrent)

    workers = [
        asyncio.create_task(
            _worker(i, queue, client, semaphore)
        )
        for i in range(app.max_concurrent)
    ]

    await asyncio.gather(*workers)
