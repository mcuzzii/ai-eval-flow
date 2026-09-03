"""
NLP Analytics Tab — async background version
===============================================

Uses background.py's submit/cancel/fragment/get_state/set_state helpers to:
  1. Kick off an async task that walks the chunk folder, calls Ollama (async HTTP)
     for each file, and computes non-LLM stats.
  2. After every chunk, append the result row to a JSON-lines file on disk
     (results.jsonl) — this is the persistence layer, decoupled from the UI.
  3. A `@fragment(run_every=...)` periodically re-reads that file and re-renders
     the dashboard, so the UI updates live without blocking on the LLM calls.

Requires:
    pip install streamlit httpx pandas plotly

Usage in your main app (background.py must be importable):

    from nlp_analytics_tab import render_nlp_analytics_tab

    tab1, tab2 = st.tabs(["Existing Tab", "NLP Analytics"])
    with tab2:
        render_nlp_analytics_tab()

Notes on integration with your project-state pattern:
  - get_state/set_state read from st.session_state[st.session_state.selected_project['id']]
  - This module stores its own bits of state under keys:
        ('nlp', 'running_future'), ('nlp', 'results_path'), ('nlp', 'config'),
        ('nlp', 'status')
  - Make sure st.session_state.selected_project is set, and that
        st.session_state[selected_project_id] = {}
    has been initialized somewhere before this tab is rendered (same as the
    rest of your app's convention).
"""

import os
import re
import json
import glob
import asyncio
from collections import Counter

from dotenv import load_dotenv

load_dotenv()

import streamlit as st
import pandas as pd
import plotly.express as px
import httpx

from background import submit, cancel, fragment, get_state, set_state


# ----------------------------------------------------------------------
# Config: controlled vocabularies
# ----------------------------------------------------------------------

DOCUMENT_TYPES = [
    "Project", "Annual Report", "Audit Report", "Overseas Program",
    "Policy Brief", "Partnerships", "Memorandum Circular", "Administrative Order",
]

DOMAINS = [
    "industry", "research", "financial sector", "market",
    "legal", "export", "finance", "review",
]

TOPICS = [
    "awaiting terminal report", "approved budget", "human resource",
    "development project", "duration", "project status", "executive summary",
    "completed projects", "steps agency actions", "processing time",
    "person responsible", "client steps agency", "technology transfer",
    "paid processing time", "terminal report", "emerging technology",
    "research", "agency actions fees", "infrastructure development",
    "program applicants", "sustained economic growth", "provided",
    "highly technical science and technology",
]


# ----------------------------------------------------------------------
# Non-LLM analytics
# ----------------------------------------------------------------------

def compute_basic_stats(text: str) -> dict:
    text = text or ""
    chars = len(text)
    words = re.findall(r"\b\w+\b", text)
    word_count = len(words)
    avg_word_len = (sum(len(w) for w in words) / word_count) if word_count else 0
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s for s in sentences if s.strip()]
    sentence_count = len(sentences)
    avg_sentence_len_words = (word_count / sentence_count) if sentence_count else 0
    lines = text.splitlines()
    line_count = len([l for l in lines if l.strip()])

    return {
        "char_count": chars,
        "word_count": word_count,
        "avg_word_length": round(avg_word_len, 3),
        "sentence_count": sentence_count,
        "avg_sentence_length_words": round(avg_sentence_len_words, 3),
        "line_count": line_count,
    }


# ----------------------------------------------------------------------
# LLM prompt + async call
# ----------------------------------------------------------------------

def build_prompt(text: str) -> str:
    doc_types = ", ".join(DOCUMENT_TYPES)
    domains = ", ".join(DOMAINS)
    topics = ", ".join(TOPICS)

    return f"""You are an information extraction system. Analyze the document chunk below and return ONLY a valid JSON object (no markdown fences, no commentary) with these exact keys:

- "entities": a list of named entities found in the text (people, organizations, locations, dates, monetary amounts, project names, agencies, etc.). Each entity should be an object with "text" and "type".
- "document_type": pick the single best match from this list: [{doc_types}]
- "domain": pick the single best match from this list: [{domains}]
- "topics": pick all topics that apply (at least one, at most 5) from this list: [{topics}]

Document chunk:
\"\"\"{text[:4000]}\"\"\"

Return ONLY the JSON object.
"""


def parse_llm_json(raw: str) -> dict:
    raw = raw.strip()
    raw = re.sub(r"^```(json)?", "", raw).strip()
    raw = re.sub(r"```$", "", raw).strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                data = {}
        else:
            data = {}

    entities = data.get("entities", [])
    if not isinstance(entities, list):
        entities = []

    topics = data.get("topics", [])
    if not isinstance(topics, list):
        topics = [topics] if topics else []

    return {
        "entities": entities,
        "document_type": data.get("document_type", ""),
        "domain": data.get("domain", ""),
        "topics": topics,
    }


async def call_ollama_async(client: httpx.AsyncClient, host: str, model: str, text: str) -> dict:
    prompt = build_prompt(text)
    resp = await client.post(
        f"{host}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "options": {"temperature": 0.0},
        },
        timeout=120.0,
    )
    resp.raise_for_status()
    data = resp.json()
    return parse_llm_json(data.get("response", ""))


# ----------------------------------------------------------------------
# File helpers
# ----------------------------------------------------------------------

def list_txt_files(folder: str):
    return sorted(glob.glob(os.path.join(folder, "*.txt")))


def read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def append_jsonl(path: str, row: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl_as_df(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# The background async task
# ----------------------------------------------------------------------

async def process_folder(folder, model, host, run_llm, max_files, results_path,
                          status_path, concurrency=4):
    """
    Walk the folder, process each file, append a JSON line to results_path
    as soon as it's done. Writes status (progress) to status_path.
    Designed to be cancellable via background.cancel().
    """
    files = list_txt_files(folder)
    if max_files:
        files = files[:max_files]

    total = len(files)

    # Fresh output file each run
    open(results_path, "w").close()

    def write_status(done, current=""):
        with open(status_path, "w", encoding="utf-8") as f:
            json.dump({"done": done, "total": total, "current": current,
                       "finished": done >= total}, f)

    write_status(0)

    sem = asyncio.Semaphore(concurrency)
    done_counter = {"n": 0}
    counter_lock = asyncio.Lock()

    async with httpx.AsyncClient() as client:

        async def process_one(path):
            fname = os.path.basename(path)
            text = read_file(path)

            row = {"filename": fname}
            row.update(compute_basic_stats(text))

            if run_llm:
                async with sem:
                    try:
                        llm_out = await call_ollama_async(client, host, model, text)
                    except Exception as e:
                        llm_out = {"entities": [], "document_type": "",
                                   "domain": "", "topics": [], "error": str(e)}
                row.update(llm_out)
            else:
                row.update({"entities": [], "document_type": "", "domain": "", "topics": []})

            append_jsonl(results_path, row)

            async with counter_lock:
                done_counter["n"] += 1
                write_status(done_counter["n"], current=fname)

        tasks = [asyncio.create_task(process_one(p)) for p in files]
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            for t in tasks:
                t.cancel()
            raise
        finally:
            write_status(done_counter["n"], current="")


# ----------------------------------------------------------------------
# Dashboard rendering
# ----------------------------------------------------------------------

def render_dashboard_body(df: pd.DataFrame):
    if df.empty:
        st.info("No results yet.")
        return

    n = len(df)
    st.markdown(f"### Live results — {n} chunk(s) processed")

    st.markdown("#### Text statistics (non-LLM)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg char length", f"{df['char_count'].mean():.0f}")
    c2.metric("Avg word count", f"{df['word_count'].mean():.0f}")
    c3.metric("Avg sentence count", f"{df['sentence_count'].mean():.1f}")
    c4.metric("Avg word length", f"{df['avg_word_length'].mean():.2f}")

    stat_cols = st.columns(3)
    with stat_cols[0]:
        fig = px.histogram(df, x="char_count", nbins=30, title="Character length distribution")
        st.plotly_chart(fig, use_container_width=True, key=f"char_hist_{n}")
    with stat_cols[1]:
        fig = px.histogram(df, x="word_count", nbins=30, title="Word count distribution")
        st.plotly_chart(fig, use_container_width=True, key=f"word_hist_{n}")
    with stat_cols[2]:
        fig = px.histogram(df, x="sentence_count", nbins=30, title="Sentence count distribution")
        st.plotly_chart(fig, use_container_width=True, key=f"sent_hist_{n}")

    st.divider()
    st.markdown("#### LLM-derived analytics")

    if "document_type" in df.columns:
        llm_df = df[df["document_type"].astype(bool)]
    else:
        llm_df = pd.DataFrame()

    if not llm_df.empty:
        colA, colB = st.columns(2)

        with colA:
            dtype_counts = llm_df["document_type"].value_counts().reset_index()
            dtype_counts.columns = ["document_type", "count"]
            fig = px.bar(dtype_counts, x="document_type", y="count", title="Document type distribution")
            st.plotly_chart(fig, use_container_width=True, key=f"dtype_{n}")

        with colB:
            domain_counts = llm_df["domain"].value_counts().reset_index()
            domain_counts.columns = ["domain", "count"]
            fig = px.bar(domain_counts, x="domain", y="count", title="Domain distribution")
            st.plotly_chart(fig, use_container_width=True, key=f"domain_{n}")

        topic_series = llm_df["topics"].explode()
        topic_series = topic_series[topic_series.astype(bool)]
        if not topic_series.empty:
            topic_counts = topic_series.value_counts().reset_index()
            topic_counts.columns = ["topic", "count"]
            fig = px.bar(
                topic_counts.sort_values("count"),
                x="count", y="topic", orientation="h",
                title="Topic frequency", height=500,
            )
            st.plotly_chart(fig, use_container_width=True, key=f"topics_{n}")

        all_entities = []
        for ents in llm_df["entities"]:
            if isinstance(ents, list):
                for e in ents:
                    if isinstance(e, dict):
                        all_entities.append((e.get("text", ""), e.get("type", "")))

        if all_entities:
            ent_df = pd.DataFrame(all_entities, columns=["text", "type"])
            colC, colD = st.columns(2)
            with colC:
                type_counts = ent_df["type"].value_counts().reset_index()
                type_counts.columns = ["entity_type", "count"]
                fig = px.bar(type_counts, x="entity_type", y="count", title="Entity type distribution")
                st.plotly_chart(fig, use_container_width=True, key=f"enttype_{n}")
            with colD:
                top_entities = ent_df["text"].value_counts().head(20).reset_index()
                top_entities.columns = ["entity", "count"]
                fig = px.bar(
                    top_entities.sort_values("count"),
                    x="count", y="entity", orientation="h",
                    title="Top 20 entities", height=500,
                )
                st.plotly_chart(fig, use_container_width=True, key=f"topent_{n}")

    st.divider()
    st.markdown("#### Per-chunk results table")
    display_df = df.copy()
    if "entities" in display_df.columns:
        display_df["entities"] = display_df["entities"].apply(
            lambda x: ", ".join(f"{e.get('text','')} ({e.get('type','')})" for e in x) if isinstance(x, list) else ""
        )
    if "topics" in display_df.columns:
        display_df["topics"] = display_df["topics"].apply(
            lambda x: ", ".join(x) if isinstance(x, list) else ""
        )
    st.dataframe(display_df, use_container_width=True, height=400)


# ----------------------------------------------------------------------
# Polling fragment — re-reads results file + status periodically
# ----------------------------------------------------------------------

@fragment(run_every=2)
def _live_dashboard_fragment():
    results_path = get_state('nlp', 'results_path')
    status_path = get_state('nlp', 'status_path')

    if not results_path or not os.path.exists(results_path):
        st.info("No run started yet.")
        return

    # Progress bar from status file
    status = {"done": 0, "total": 0, "current": "", "finished": False}
    if status_path and os.path.exists(status_path):
        try:
            with open(status_path, "r", encoding="utf-8") as f:
                status = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

    total = status.get("total", 0)
    done = status.get("done", 0)
    if total:
        st.progress(min(done / total, 1.0),
                     text=f"Processed {done}/{total}" +
                          (f" — last: {status.get('current','')}" if status.get('current') else ""))
        if status.get("finished"):
            st.success(f"Finished processing {total} chunks.")

    df = load_jsonl_as_df(results_path)
    render_dashboard_body(df)


# ----------------------------------------------------------------------
# Main tab renderer
# ----------------------------------------------------------------------

def render_nlp_analytics_tab():
    st.header("NLP Analytics Dashboard")

    with st.expander("Settings", expanded=True):
        folder = get_state('project_dir') / 'chunks'
        model = 'qwen3:8b'
        host = os.environ("OLLAMA_IP_ADDRESS")
        max_files = st.number_input("Max files to process (0 = all)", min_value=0, value=0, step=50)
        concurrency = 1
        run_llm = st.checkbox("Run LLM analytics (entities, doc type, domain, topics)", value=True)

    results_path = get_state('project_dir') / "nlp_results.jsonl"
    status_path = get_state('project_dir') / "nlp_status.json"

    col1, col2 = st.columns(2)
    start_clicked = col1.button("Start analysis", type="primary")
    stop_clicked = col2.button("Stop")

    if start_clicked:
        if not os.path.isdir(folder):
            st.error(f"Folder not found: {folder}")
        else:
            # cancel any previous run
            prev_future = get_state('nlp', 'future')
            if prev_future is not None and not prev_future.done():
                cancel(prev_future)

            set_state('nlp', 'results_path', value=results_path)
            set_state('nlp', 'status_path', value=status_path)

            future = submit(process_folder(
                folder=folder,
                model=model,
                host=host,
                run_llm=run_llm,
                max_files=max_files,
                results_path=results_path,
                status_path=status_path,
                concurrency=concurrency,
            ))
            set_state('nlp', 'future', value=future)
            st.success("Started background processing — dashboard below updates automatically.")

    if stop_clicked:
        future = get_state('nlp', 'future')
        if future is not None and not future.done():
            cancel(future)
            st.warning("Cancellation requested.")
        else:
            st.info("No running task to stop.")

    st.divider()

    # Make sure paths are registered even if user reloads without clicking start
    if get_state('nlp', 'results_path') is None:
        set_state('nlp', 'results_path', value=results_path)
        set_state('nlp', 'status_path', value=status_path)

    _live_dashboard_fragment()


# Standalone testing
if __name__ == "__main__":
    st.set_page_config(layout="wide")
    if "selected_project" not in st.session_state:
        st.session_state.selected_project = {"id": "default"}
    if st.session_state.selected_project["id"] not in st.session_state:
        st.session_state[st.session_state.selected_project["id"]] = {}
    render_nlp_analytics_tab()