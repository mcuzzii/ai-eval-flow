import streamlit as st
import streamlit.components.v1 as components
import threading
import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
import sys
sys.path.append(str(Path.cwd() / 'src'))
import pandas as pd
import uuid
import queue
from background import submit, cancel, get_state
from state import AppState, Pair, STATUS_PENDING, STATUS_RUNNING, \
    STATUS_APPROVED, STATUS_FLAGGED, STATUS_EDITED, STATUS_SKIPPED

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AIEvalFlow",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ────────────────────────────────────────────────────────────────

# app.py
def load_css(path: str):
    with open(path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("src/styles.css")

# ── init shared state ─────────────────────────────────────────────────────────
if "app_state" not in st.session_state:
    st.session_state.app_state = AppState()

if "selected_idx" not in st.session_state:
    st.session_state.selected_idx = None

if "selected_project" not in st.session_state:
    st.session_state.selected_project = None

if "filter" not in st.session_state:
    st.session_state.filter = "flagged"

if "active_tab" not in st.session_state:
    st.session_state.active_tab = "home"

if "projects_dir" not in st.session_state:
    st.session_state.projects_dir = Path('.cache') / 'projects'
    st.session_state.projects_dir.mkdir(exist_ok=True, parents=True)

if "metadata_list" not in st.session_state:
    st.session_state.metadata_list = list()

if "project_dirs" not in st.session_state:
    st.session_state.project_dirs = list()

    for project_dir in Path('.cache/projects').iterdir():
        if project_dir.is_dir():
            st.session_state.project_dirs.append(project_dir)

            with open(project_dir / 'desc.json', 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            st.session_state.metadata_list.append(metadata)

            if metadata['id'] not in st.session_state:
                st.session_state[metadata['id']] = {
                    'active_dialog': None,
                    'project_dir': project_dir,
                    'sources': dict(),
                    'docs': dict(),
                    'chunks': dict(),
                    'upload_queue': queue.Queue(),
                    'displayed_docs': set(),
                    'displayed_chunks': set(),
                    'sources_cache_loaded': threading.Event(),
                    'docs_cache_loaded': threading.Event(),
                    'chunks_cache_loaded': threading.Event(),
                    'sources_load_started': False,
                    'docs_load_started': False,
                    'chunks_load_started': False,
                    'coro_queue': queue.Queue(),
                    'running_coroutines': {
                        'uploading': list(),
                        'caching': list(),
                        'chunking': list()
                    },
                    'cache_queue': queue.Queue(),
                    'chunk_queue': queue.Queue(),
                    'delete_queue': queue.Queue(),
                    'active_source': None,
                    'display_queue': queue.Queue(),
                    'display_key': None,
                    'display_text': "Loading content..."
                }

from documents import render_documents_tab
from critic import run_critics
from templates import score_color, status_badge, metric_card

app: AppState = st.session_state.app_state

# ── helpers ───────────────────────────────────────────────────────────────────

def start_workers():
    if app.worker_started:
        return
    def _run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(run_critics(app))
    t = threading.Thread(target=_run, daemon=True)
    t.start()
    app.worker_started = True

def write_metadata(project_dir, metadata):
    with open(project_dir / 'desc.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
def create_or_edit_project(project_dir, metadata, name, desc, confirm_label):
    existing_names = [p["name"] for p in st.session_state.metadata_list]

    error_message = None

    col1, col2 = st.columns(2)
    with col1:
        if st.button(confirm_label, type="primary", use_container_width=True):
            if not name:
                error_message = "Project name is required"
            elif not metadata and name in existing_names:
                error_message = f"A project named '{name}' already exists"
            else:
                new_project_metadata = {
                    "name": name,
                    "description": desc,
                    "id": metadata.get('id') if metadata else str(uuid.uuid4())
                }
                if not project_dir:
                    project_dir = st.session_state.projects_dir / name
                    project_dir.mkdir(exist_ok=True)
                write_metadata(project_dir, new_project_metadata)
                st.rerun()
    with col2:
        if st.button("Cancel", type="tertiary", use_container_width=True):
            st.rerun()
    
    if error_message:
        st.error(error_message)

@st.dialog("Create new project")
def create_project_dialog():
    name = st.text_input("Project Name")
    desc = st.text_area("Description")

    create_or_edit_project(None, None, name, desc, "Create")

@st.dialog("Edit Project")
def edit_project_dialog(project_dir, metadata):
    name = st.text_input("Project Name", value=metadata.get('name'))
    desc = st.text_area("Description", value=metadata.get('description'))
    
    create_or_edit_project(project_dir, metadata, name, desc, "Save")

@st.dialog("Delete project")
def delete_project(project_dir: Path):
    st.markdown(
        'Are you sure you want to delete this project? This action cannot be undone.',
        unsafe_allow_html=True
    )
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Yes", type='secondary', width='stretch'):
            for item in project_dir.iterdir():
                item.unlink()
            project_dir.rmdir()
            st.rerun()
    with col2:
        if st.button("Cancel", type='primary', width='stretch'):
            st.rerun()

def load_tabs(tabs, global_tabs):
    for key, label in tabs.items():
        if st.button(label, width='stretch',
                     type="primary" if st.session_state.active_tab == key else "secondary"):
            if global_tabs:
                st.session_state.selected_project = None
            st.session_state.active_tab = key
            st.rerun()

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔬 AIEvalFlow")
    st.markdown("---")

    # file upload
    # uploaded = st.file_uploader("Load QA pairs (JSON)", type="json", key="upload")
    # if uploaded and not app.loaded:
    #     raw = json.load(uploaded)
    #     app.load_pairs(raw)
    #     st.success(f"Loaded {len(app.pairs)} pairs")

    # if app.loaded and not app.worker_started:
    #     if st.button("▶  Start critiquing", use_container_width=True):
    #         start_workers()
    #         st.rerun()

    # if app.worker_started:
    #     st.markdown(f"**Workers running** ({app.max_concurrent} parallel)")

    # st.markdown("---")

    # nav
    tabs = {
        "home": "Home",
        "projects": "Projects",
    }
    load_tabs(tabs, global_tabs=True)

    st.markdown("---")

    if st.session_state.selected_project:
        st.caption(f"CURRENT PROJECT")
        st.markdown(f"""
        <div class="sidebar-project-card">
            <div class="sidebar-project-name">{st.session_state.selected_project['name']}</div>
            <div class="sidebar-project-status-idle">Idle</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        project_tabs = {
            "overview": "Overview",
            "documents": "Documents",
            "qa_pairs": "Q&A Pairs",
            "metrics": "Metrics"
        }
        load_tabs(project_tabs, global_tabs=False)


    # if app.loaded:
    #     if st.button("💾 Save output", use_container_width=True):
    #         path = "qa_pairs_refined.json"
    #         app.save(path)
    #         st.success(f"Saved to {path}")

    # st.markdown("---")
    # st.caption("Critic model: `qwen3.6:35b`")
    # st.caption("Rewrite model: `nemotron3:33b`")

# ══════════════════════════════════════════════════════════════════════════════
# LANDING PAGE
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.active_tab == "home":
 
    # ── Hero ──────────────────────────────────────────────────────────────────
    st.markdown('<span class="badge">AIEvalFlow - v1.0</span>', unsafe_allow_html=True)
    st.title("Know where your chatbot fails, before your users do.")
    st.markdown("A structured quality assurance workflow to measure chatbot response quality at different steps of the pipeline.")
 
    col1, col2, col3 = st.columns([1.25, 1.25, 4])
    with col1:
        new_project = st.button(
            "+ New project",
            type="primary",
            width='stretch'
        )
    with col2:
        open_project = st.button(
            "📂 Open project",
            width='stretch'
        )
    with col3:
        demo_clicked = st.button(
            "▶ Run demo",
            width='content',
            disabled=True,
            help="Coming soon"
        )
 
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
 
    # ── Pipeline diagram ──────────────────────────────────────────────────────
    st.caption("HOW IT WORKS")
    st.markdown("""
    <div class="pipe-container">
    <div class="pipe-step">
        <div class="pipe-icon">📥</div>
        <div class="pipe-label">Input</div>
        <div class="pipe-desc">Load reference ground-truth documents</div>
    </div>
    <div class="pipe-arrow">→</div>
    <div class="pipe-step">
        <div class="pipe-icon">🗂️</div>
        <div class="pipe-label">Preprocess</div>
        <div class="pipe-desc">Generate and analyze document chunks</div>
    </div>
    <div class="pipe-arrow">→</div>
    <div class="pipe-step">
        <div class="pipe-icon">📝</div>
        <div class="pipe-label">Synthesize</div>
        <div class="pipe-desc">Create ground-truth Q&ampA pairs using an LLM</div>
    </div>
    <div class="pipe-arrow">→</div>
    <div class="pipe-step">
        <div class="pipe-icon">✍️</div>
        <div class="pipe-label">Refine</div>
        <div class="pipe-desc">Edit ground-truth Q&ampA pairs with LLM feedback</div>
    </div>
    <div class="pipe-arrow">→</div>
    <div class="pipe-step">
        <div class="pipe-icon">🤖</div>
        <div class="pipe-label">Generate</div>
        <div class="pipe-desc">Run ground-truth questions through the test chatbot</div>
    </div>
    <div class="pipe-arrow">→</div>
    <div class="pipe-step">
        <div class="pipe-icon">📐</div>
        <div class="pipe-label">Evaluate</div>
        <div class="pipe-desc">Compare test chatbot responses to ground-truth answers with a judge LLM</div>
    </div>
    <div class="pipe-arrow">→</div>
    <div class="pipe-step">
        <div class="pipe-icon">📊</div>
        <div class="pipe-label">Report</div>
        <div class="pipe-desc">View scores, trends, &amp failures</div>
    </div>
    </div>
    """, unsafe_allow_html=True)
 
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Sample results preview ────────────────────────────────────────────────
    st.caption("SAMPLE RESULTS")
    preview = [
        ("What is the refund policy for enterprise plans?", 0.91, "✅ Pass"),
        ("Can you summarize last quarter's sales report?", 0.74, "⚠️ Review"),
        ("How do I escalate a billing dispute?", 0.41, "❌ Fail"),
    ]
    df = pd.DataFrame(preview, columns=["Prompt", "Score", "Status"])
    st.dataframe(df, width='stretch', hide_index=True)
 
    # ── Run demo ──────────────────────────────────────────────────────────────
    if new_project:
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)


# ── PROJECTS TAB ───────────────────────────────────────────────────────
if st.session_state.active_tab == 'projects':
    st.markdown('<span class="badge">AIEvalFlow - v1.0</span>', unsafe_allow_html=True)
    st.title("Projects")

    col_caption, col_new = st.columns([1, 1])
    with col_caption:
        st.markdown('Start a new or open an existing project')
    with col_new:
        if st.button("+ New project", width='stretch', type='secondary'):
            create_project_dialog()
    
    for project_dir, metadata in zip(st.session_state.project_dirs, st.session_state.metadata_list):

        name = metadata.get('name')
        desc = metadata.get('description')
        key = metadata.get('id')

        with st.container(border=True):
            st.markdown(f"""
            <div class="project-name">{name}</div>
            <div class="project-desc">{desc}</div>
            """, unsafe_allow_html=True)
            col_open, col_edit, col_run, col_stop, col_del = st.columns([4.75, 1.25, 1, 1, 1])
            with col_open:
                if st.button("Open", width='stretch', type='secondary', key=f'{key}_open'):
                    st.session_state.selected_project = {
                        "dir": project_dir,
                        **metadata
                    }
                    st.session_state.active_tab = "overview"
                    st.rerun()
            
            with col_edit:
                if st.button("✏️ Edit", width='stretch', type='secondary', key=f'{key}_edit'):
                    edit_project_dialog(project_dir, metadata)

            with col_run:
                if st.button("▶ Run", width='stretch',  type='primary', key=f'{key}_run'):
                    st.write("hi!")
            
            with col_stop:
                if st.button("⏹ Stop", width='stretch', type='tertiary', key=f'{key}_stop'):
                    st.write("Hey!")

            with col_del:
                if st.button("🗑 Del", width='stretch', type='tertiary', key=f'{key}_del'):
                    delete_project(project_dir)

# —— DOCUMENTS TAB ─────────────────────────────────────────────────────
if st.session_state.active_tab == "documents":
    st.markdown('<span class="badge">AIEvalFlow - v1.0</span>', unsafe_allow_html=True)
    render_documents_tab()

# ── CRITIQUE & EDIT TAB ───────────────────────────────────────────────────────
if st.session_state.active_tab == "qa_pairs":

    if not app.worker_started:
         if st.button("▶  Start critiquing", use_container_width=True):
            qa_pairs_path = get_state('project_dir') / 'qa.json'
            with open(qa_pairs_path, 'r', encoding='utf-8') as f:
                qa_pairs = json.load(f)
            app.load_pairs(qa_pairs)
            start_workers()
            st.rerun()

    if app.worker_started:
        st.markdown(f"**Workers running** ({app.max_concurrent} parallel)")

    # stats row
    total    = len(app.pairs)
    approved = sum(1 for p in app.pairs if p.status in (STATUS_APPROVED, STATUS_EDITED))
    flagged  = sum(1 for p in app.pairs if p.status == STATUS_FLAGGED)
    pending  = sum(1 for p in app.pairs if p.status == STATUS_PENDING)
    running  = sum(1 for p in app.pairs if p.status == STATUS_RUNNING)
    done     = approved + sum(1 for p in app.pairs if p.status in (STATUS_SKIPPED,))

    cols = st.columns(5)
    with cols[0]: st.markdown(metric_card("Total", total, "gray"), unsafe_allow_html=True)
    with cols[1]: st.markdown(metric_card("Approved", approved, "green"), unsafe_allow_html=True)
    with cols[2]: st.markdown(metric_card("Flagged", flagged, "yellow"), unsafe_allow_html=True)
    with cols[3]: st.markdown(metric_card("Running", running, "blue"), unsafe_allow_html=True)
    with cols[4]: st.markdown(metric_card("Pending", pending, "gray"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # progress bar
    pct = int((done + flagged) / total * 100) if total else 0
    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:12px; margin-bottom:1.5rem;">
        <div class="progress-track" style="flex:1">
            <div class="progress-fill" style="width:{pct}%"></div>
        </div>
        <span style="font-family:'IBM Plex Mono',monospace; font-size:12px; color:#555; white-space:nowrap">
            {done + flagged} / {total}
        </span>
    </div>
    """, unsafe_allow_html=True)

    # two-column layout: list | detail
    left, right = st.columns([1, 1.6], gap="large")

    with left:
        st.markdown("#### Pairs")

        # filter tabs
        filter_opts = ["flagged", "all", "approved", "pending"]
        fcols = st.columns(len(filter_opts))
        for i, f in enumerate(filter_opts):
            with fcols[i]:
                if st.button(f, key=f"filter_{f}", use_container_width=True,
                             type="primary" if st.session_state.filter == f else "secondary"):
                    st.session_state.filter = f
                    st.rerun()

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        # filtered list
        filt = st.session_state.filter
        shown = [p for p in app.pairs if filt == "all" or p.status == filt
                 or (filt == "flagged" and p.status == STATUS_FLAGGED)]

        if not shown:
            st.caption(f"No pairs with status '{filt}'")
        else:
            for p in shown[:80]:  # cap at 80 for perf
                c = p.critique
                avg_str = ""
                if c:
                    avg = (c["accuracy"] + c["completeness"] + c["clarity"]) / 3
                    avg_str = f" · avg {avg:.1f}"

                selected = st.session_state.selected_idx == p.idx
                label = f"[{p.idx:>3}] {p.question[:45]}{'...' if len(p.question)>45 else ''}"
                sublabel = f"{p.status}{avg_str}"

                if st.button(f"{label}\n{sublabel}", key=f"pair_{p.idx}",
                             use_container_width=True,
                             type="primary" if selected else "secondary"):
                    st.session_state.selected_idx = p.idx
                    st.rerun()

    with right:
        idx = st.session_state.selected_idx
        if idx is None:
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.caption("← Select a pair to review")
        else:
            p: Pair = app.pairs[idx]

            st.markdown(f"#### Pair {idx} &nbsp; {status_badge(p.status)}",
                        unsafe_allow_html=True)

            st.markdown('<div class="section-label">Question</div>', unsafe_allow_html=True)
            st.markdown(f"> {p.question}")

            st.markdown('<div class="section-label">Reference chunk</div>',
                        unsafe_allow_html=True)
            ref_preview = p.reference[:600] + ("..." if len(p.reference) > 600 else "")
            st.markdown(f'<div class="ref-block">{ref_preview}</div>',
                        unsafe_allow_html=True)

            st.markdown('<div class="section-label">Original answer</div>',
                        unsafe_allow_html=True)
            st.markdown(f'<div class="answer-block">{p.answer}</div>',
                        unsafe_allow_html=True)

            # critique
            if p.critique:
                c = p.critique
                st.markdown('<div class="section-label">Critique</div>',
                            unsafe_allow_html=True)

                score_html = "".join([
                    f'<span class="score-pill {score_color(c["accuracy"])}">acc {c["accuracy"]}</span>',
                    f'<span class="score-pill {score_color(c["completeness"])}">comp {c["completeness"]}</span>',
                    f'<span class="score-pill {score_color(c["clarity"])}">clarity {c["clarity"]}</span>',
                ])
                st.markdown(score_html, unsafe_allow_html=True)

                if c.get("issues"):
                    for issue in c["issues"]:
                        st.markdown(f'<div class="issue-item">⚠ {issue}</div>',
                                    unsafe_allow_html=True)
                else:
                    st.caption("No issues found")

            if p.status == STATUS_RUNNING:
                st.info("Critiquing in progress...")

            # edit box — only show for flagged pairs
            if p.status == STATUS_FLAGGED:
                st.markdown('<div class="section-label">Your edit</div>',
                            unsafe_allow_html=True)
                new_answer = st.text_area(
                    label="",
                    value=p.final_answer,
                    height=180,
                    key=f"edit_{idx}",
                    placeholder="Edit the answer here, then click Save edit",
                    label_visibility="collapsed",
                )

                bcols = st.columns(3)
                with bcols[0]:
                    if st.button("✅ Save edit", key=f"save_{idx}",
                                 use_container_width=True, type="primary"):
                        p.final_answer = new_answer
                        p.status = STATUS_EDITED
                        st.rerun()
                with bcols[1]:
                    if st.button("👍 Approve as-is", key=f"approve_{idx}",
                                 use_container_width=True):
                        p.status = STATUS_APPROVED
                        st.rerun()
                with bcols[2]:
                    if st.button("⏭ Skip", key=f"skip_{idx}",
                                 use_container_width=True):
                        p.status = STATUS_SKIPPED
                        st.rerun()

            elif p.status in (STATUS_EDITED, STATUS_APPROVED):
                st.markdown('<div class="section-label">Final answer</div>',
                            unsafe_allow_html=True)
                st.markdown(f'<div class="answer-block refined">{p.final_answer}</div>',
                            unsafe_allow_html=True)

                if st.button("✏️ Re-edit", key=f"reedit_{idx}"):
                    p.status = STATUS_FLAGGED
                    st.rerun()

# ── METRICS TAB ───────────────────────────────────────────────────────────────
elif st.session_state.active_tab == "metrics":
    import pandas as pd

    st.markdown("#### Critique score distribution")

    scored = [p for p in app.pairs if p.critique]
    if not scored:
        st.info("No pairs have been critiqued yet.")
    else:
        rows = []
        for p in scored:
            c = p.critique
            rows.append({
                "idx":          p.idx,
                "question":     p.question[:60],
                "status":       p.status,
                "accuracy":     c["accuracy"],
                "completeness": c["completeness"],
                "clarity":      c["clarity"],
                "avg":          round((c["accuracy"] + c["completeness"] + c["clarity"]) / 3, 2),
                "needs_edit":   c.get("needs_edit", False),
            })

        df = pd.DataFrame(rows)

        

        # summary metrics
        mcols = st.columns(4)
        with mcols[0]:
            st.metric("Avg accuracy",     f"{df['accuracy'].mean():.2f} / 5")
        with mcols[1]:
            st.metric("Avg completeness", f"{df['completeness'].mean():.2f} / 5")
        with mcols[2]:
            st.metric("Avg clarity",      f"{df['clarity'].mean():.2f} / 5")
        with mcols[3]:
            st.metric("Flagged rate",
                      f"{df['needs_edit'].sum()} / {len(df)} "
                      f"({int(df['needs_edit'].mean()*100)}%)")

        st.markdown("---")

        # score histograms
        hcols = st.columns(3)
        for col, dim in zip(hcols, ["accuracy", "completeness", "clarity"]):
            with col:
                st.markdown(f"**{dim.capitalize()}**")
                counts = df[dim].value_counts().sort_index()
                st.bar_chart(counts)

        st.markdown("---")
        st.markdown("#### All scored pairs")

        status_filter = st.multiselect(
            "Filter by status",
            options=df["status"].unique().tolist(),
            default=df["status"].unique().tolist(),
        )

        filtered_df = df[df["status"].isin(status_filter)]
        st.dataframe(
            filtered_df[["idx", "question", "status", "accuracy",
                          "completeness", "clarity", "avg"]],
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "⬇ Download scores CSV",
            data=filtered_df.to_csv(index=False),
            file_name="qa_scores.csv",
            mime="text/csv",
        )

# ── auto-refresh while workers are running ────────────────────────────────────
if app.worker_started and any(p.status in (STATUS_RUNNING, STATUS_PENDING)
                               for p in app.pairs):
    time.sleep(1.5)
    st.rerun()
