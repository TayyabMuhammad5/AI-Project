"""
RACE Reading Comprehension — Streamlit UI
Run:  streamlit run UI/app.py   (from project root)
"""

import os, sys, time, random, io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ── path fix so components.py can find src/ ──
_this = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_this, '..'))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, _this)

# ─────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="RACE · AI Quiz Engine",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

:root {
    --bg-primary: #0f1117;
    --bg-card: #1a1d29;
    --bg-card-hover: #22263a;
    --accent: #6c63ff;
    --accent-light: #8b83ff;
    --green: #00d68f;
    --red: #ff6b6b;
    --yellow: #ffc107;
    --text-primary: #e8e8f0;
    --text-secondary: #8b8da3;
    --border: #2a2d3e;
}

html, body, [class*="st-"] {
    font-family: 'Inter', sans-serif !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #12141f 0%, #1a1d2e 100%);
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] .stRadio label {
    font-size: 15px !important;
    padding: 8px 4px !important;
}

/* Cards */
.card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 24px;
    margin-bottom: 16px;
    transition: transform .2s, box-shadow .2s;
}
.card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(108,99,255,.12);
}

/* Option buttons */
.opt-btn {
    display: block;
    width: 100%;
    text-align: left;
    padding: 14px 20px;
    margin: 8px 0;
    border-radius: 12px;
    border: 2px solid var(--border);
    background: var(--bg-card);
    color: var(--text-primary);
    font-size: 15px;
    cursor: pointer;
    transition: all .2s;
}
.opt-btn:hover { border-color: var(--accent); background: var(--bg-card-hover); }
.opt-correct { border-color: var(--green) !important; background: rgba(0,214,143,.08) !important; }
.opt-wrong   { border-color: var(--red) !important; background: rgba(255,107,107,.08) !important; }

/* Stat pill */
.stat-pill {
    display: inline-block;
    background: linear-gradient(135deg, var(--accent), #8b5cf6);
    color: white;
    padding: 6px 16px;
    border-radius: 20px;
    font-weight: 600;
    font-size: 13px;
    margin: 2px 4px;
}

/* Hint boxes */
.hint-box {
    background: linear-gradient(135deg, #1e2235, #252a42);
    border-left: 4px solid var(--yellow);
    border-radius: 0 12px 12px 0;
    padding: 16px 20px;
    margin: 10px 0;
    font-size: 14px;
}
.hint-label {
    color: var(--yellow);
    font-weight: 700;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 6px;
}

/* Hero header */
.hero {
    text-align: center;
    padding: 30px 0 10px;
}
.hero h1 {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #6c63ff, #00d68f);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 4px;
}
.hero p { color: var(--text-secondary); font-size: 1rem; }

/* Metric card */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 20px;
    text-align: center;
}
.metric-card .value {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #6c63ff, #00d68f);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.metric-card .label { color: var(--text-secondary); font-size: .85rem; margin-top: 4px; }

/* Scrollable article area */
.article-scroll {
    max-height: 280px;
    overflow-y: auto;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 18px;
    font-size: 14px;
    line-height: 1.7;
    color: var(--text-primary);
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────
# LOAD BACKEND (cached)
# ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading ML models … please wait ⏳")
def load_backend():
    import components as comp
    return comp

comp = load_backend()


# ─────────────────────────────────────────────────────
# SESSION STATE DEFAULTS
# ─────────────────────────────────────────────────────
defaults = {
    'page': '📝 Article Input',
    'article': '', 'question': '', 'choices': ['','','',''],
    'correct_idx': None, 'correct_label': None,
    'quiz_ready': False, 'user_answer': None, 'checked': False,
    'hints_revealed': 0, 'hints_data': [],
    'model_results': None, 'latency': 0.0,
    'inference_log': [],
    'generated_q': False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 RACE Quiz Engine")
    st.caption("AI-powered reading comprehension")
    st.divider()
    page = st.radio(
        "Navigate",
        ["📝 Article Input", "❓ Quiz View", "💡 Hint Panel", "📊 Dashboard"],
        index=["📝 Article Input", "❓ Quiz View", "💡 Hint Panel", "📊 Dashboard"].index(st.session_state['page']),
        key='nav_radio',
    )
    st.session_state['page'] = page
    st.divider()
    st.caption(f"Session inferences: **{len(st.session_state['inference_log'])}**")
    if st.button("🔄 Reset Session", use_container_width=True):
        for k, v in defaults.items():
            st.session_state[k] = v
        st.rerun()


# ═════════════════════════════════════════════════════
# PAGE 1 — ARTICLE INPUT
# ═════════════════════════════════════════════════════
if page == "📝 Article Input":
    st.markdown('<div class="hero"><h1>📝 Article Input</h1><p>Paste a passage or load a RACE sample, then generate a quiz!</p></div>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col2:
        st.markdown("#### Quick Actions")
        if st.button("🎲 Load Random RACE Sample", use_container_width=True):
            with st.spinner("Sampling from RACE dataset…"):
                sample = comp.load_random_race_sample()
            st.session_state['article'] = sample['article']
            st.session_state['article_input'] = sample['article']
            
            st.session_state['question'] = sample['question']
            st.session_state['question_input'] = sample['question']
            
            st.session_state['choices'] = sample['choices']
            for i in range(4):
                st.session_state[f'opt_{i}'] = sample['choices'][i]
                
            st.session_state['correct_idx'] = sample['correct_idx']
            st.session_state['correct_label'] = sample['correct_label']
            st.session_state['correct_idx_input'] = sample['correct_label']
            
            st.session_state['quiz_ready'] = False
            st.session_state['checked'] = False
            st.session_state['user_answer'] = None
            st.session_state['hints_revealed'] = 0
            st.session_state['generated_q'] = False
            st.rerun()

        st.markdown("---")
        st.info("💡 **Tip:** You can paste your own article and write your own question + choices, or let the AI generate them!")

    with col1:
        article = st.text_area(
            "📖 Reading Passage",
            value=st.session_state['article'],
            height=220,
            placeholder="Paste your article here…",
            key='article_input',
        )
        st.session_state['article'] = article

    if not article.strip():
        st.warning("⬆️ Paste an article above or click **Load Random RACE Sample** to get started.")
        st.stop()

    st.divider()

    # Question + choices input
    st.markdown("### ❓ Question & Choices")
    mode = st.radio("How would you like to set up the quiz?",
                     ["✏️ Enter manually", "🤖 Auto-generate (AI)"],
                     horizontal=True)

    if mode == "✏️ Enter manually":
        q = st.text_input("Question", value=st.session_state['question'], placeholder="Type your question…", key='question_input')
        st.session_state['question'] = q
        cols = st.columns(4)
        labels = ['A', 'B', 'C', 'D']
        for i, c in enumerate(cols):
            with c:
                val = st.text_input(f"Option {labels[i]}", value=st.session_state['choices'][i] if i < len(st.session_state['choices']) else '', key=f'opt_{i}')
                st.session_state['choices'][i] = val
        ci = st.selectbox("Correct answer", labels, index=st.session_state['correct_idx'] if st.session_state['correct_idx'] is not None else 0, key='correct_idx_input')
        st.session_state['correct_idx'] = labels.index(ci)
        st.session_state['correct_label'] = ci

    else:  # auto-generate
        if st.button("⚡ Generate Question + Distractors", use_container_width=True, type="primary"):
            with st.spinner("Generating question…"):
                # pick a key sentence as the "answer"
                from hint_generator import split_sentences, content_words
                sents = split_sentences(article)
                if not sents:
                    st.error("Article is too short to generate a question.")
                    st.stop()
                # Use a mid-length sentence as answer proxy
                mid = sents[min(1, len(sents)-1)]
                words = mid.split()
                answer_phrase = ' '.join(words[:min(5, len(words))])

                gen_q = comp.generate_question(article, answer_phrase)
                distractors = comp.generate_distractors(article, gen_q, answer_phrase, top_k=3)

                all_choices = [answer_phrase] + distractors
                random.shuffle(all_choices)
                correct_idx = all_choices.index(answer_phrase)

                st.session_state['question'] = gen_q
                st.session_state['choices'] = all_choices
                st.session_state['correct_idx'] = correct_idx
                st.session_state['correct_label'] = ['A','B','C','D'][correct_idx]
                st.session_state['generated_q'] = True
            st.rerun()

        if st.session_state['generated_q']:
            st.success(f"✅ Generated question: **{st.session_state['question']}**")
            for i, ch in enumerate(st.session_state['choices']):
                tag = " ✓" if i == st.session_state['correct_idx'] else ""
                st.write(f"**{['A','B','C','D'][i]})** {ch}{tag}")

    # Submit
    st.divider()
    can_submit = (article.strip() and st.session_state['question'].strip()
                  and any(c.strip() for c in st.session_state['choices']))
    if st.button("🚀 Submit & Start Quiz", use_container_width=True, type="primary", disabled=not can_submit):
        with st.spinner("Running Model A inference…"):
            t0 = time.time()
            results = comp.predict_answer(
                st.session_state['article'],
                st.session_state['question'],
                st.session_state['choices'],
            )
            latency = time.time() - t0
        st.session_state['model_results'] = results
        st.session_state['latency'] = latency
        st.session_state['quiz_ready'] = True
        st.session_state['checked'] = False
        st.session_state['user_answer'] = None
        st.session_state['hints_revealed'] = 0
        # Generate hints
        correct_text = st.session_state['choices'][st.session_state['correct_idx']] if st.session_state['correct_idx'] is not None else ''
        st.session_state['hints_data'] = comp.generate_hints(
            st.session_state['article'], st.session_state['question'], correct_text
        )
        st.session_state['page'] = "❓ Quiz View"
        st.rerun()


# ═════════════════════════════════════════════════════
# PAGE 2 — QUIZ VIEW
# ═════════════════════════════════════════════════════
elif page == "❓ Quiz View":
    if not st.session_state['quiz_ready']:
        st.markdown('<div class="hero"><h1>❓ Quiz View</h1><p>No quiz loaded yet. Go to Article Input first.</p></div>', unsafe_allow_html=True)
        if st.button("← Go to Article Input"):
            st.session_state['page'] = "📝 Article Input"
            st.rerun()
        st.stop()

    st.markdown('<div class="hero"><h1>❓ Quiz Time!</h1><p>Read the passage and pick the correct answer</p></div>', unsafe_allow_html=True)

    # Show article in a scrollable card
    st.markdown(f'<div class="article-scroll">{st.session_state["article"][:2000]}</div>', unsafe_allow_html=True)
    st.markdown("")

    # Question
    st.markdown(f"### 📌 {st.session_state['question']}")

    labels = ['A', 'B', 'C', 'D']
    choices = st.session_state['choices']
    correct_idx = st.session_state['correct_idx']

    # Answer selection
    if not st.session_state['checked']:
        selected = st.radio(
            "Choose your answer:",
            [f"**{labels[i]})** {choices[i]}" for i in range(len(choices))],
            index=None,
            key='quiz_select',
        )
        if selected:
            st.session_state['user_answer'] = [f"**{labels[i]})** {choices[i]}" for i in range(len(choices))].index(selected)

        if st.button("✅ Check Answer", use_container_width=True, type="primary",
                      disabled=st.session_state['user_answer'] is None):
            st.session_state['checked'] = True
            # Log inference
            results = st.session_state['model_results']
            log_entry = {
                'y_true': labels[correct_idx] if correct_idx is not None else 'A',
                'y_pred_lr': results['Logistic Regression']['label'],
                'y_pred_svm': results['SVM']['label'],
                'y_pred_km': results['K-Means']['label'],
                'y_pred_ens': results['Ensemble (Majority Vote)']['label'],
                'latency': st.session_state['latency'],
                'question': st.session_state['question'][:60],
            }
            st.session_state['inference_log'].append(log_entry)
            
            st.rerun()
    else:
        # Show results
        user_idx = st.session_state['user_answer']
        is_correct = (user_idx == correct_idx)

        for i, ch in enumerate(choices):
            if i == correct_idx:
                st.markdown(f'<div class="opt-btn opt-correct">✅ <b>{labels[i]})</b> {ch}</div>', unsafe_allow_html=True)
            elif i == user_idx and not is_correct:
                st.markdown(f'<div class="opt-btn opt-wrong">❌ <b>{labels[i]})</b> {ch}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="opt-btn"><b>{labels[i]})</b> {ch}</div>', unsafe_allow_html=True)

        if is_correct:
            st.success("🎉 **Correct!** Great job!")
        else:
            st.error(f"❌ **Incorrect.** The right answer was **{labels[correct_idx]}) {choices[correct_idx]}**")

        # Model predictions comparison
        st.divider()
        st.markdown("### 🤖 Model Predictions")
        results = st.session_state['model_results']
        mcols = st.columns(len(results))
        for i, (model_name, res) in enumerate(results.items()):
            with mcols[i]:
                icon = "✅" if res['label'] == labels[correct_idx] else "❌"
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size:.8rem;color:var(--text-secondary);">{model_name}</div>
                    <div class="value">{res['label']}</div>
                    <div class="label">{icon} {res['text'][:40]}</div>
                </div>
                """, unsafe_allow_html=True)

        # Score details in expander
        with st.expander("📊 Detailed Model Scores"):
            for model_name, res in results.items():
                st.markdown(f"**{model_name}** — _{res['score_type']}_")
                score_df = pd.DataFrame([res['scores']]).T.rename(columns={0:'Score'})
                st.dataframe(score_df, use_container_width=True)

        st.markdown(f'<div class="stat-pill">⏱ Latency: {st.session_state["latency"]:.3f}s</div>', unsafe_allow_html=True)

        st.markdown("")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💡 View Hints", use_container_width=True):
                st.session_state['page'] = "💡 Hint Panel"
                st.rerun()
        with col2:
            if st.button("🔁 New Question", use_container_width=True):
                st.session_state['quiz_ready'] = False
                st.session_state['checked'] = False
                st.session_state['page'] = "📝 Article Input"
                st.rerun()


# ═════════════════════════════════════════════════════
# PAGE 3 — HINT PANEL
# ═════════════════════════════════════════════════════
elif page == "💡 Hint Panel":
    st.markdown('<div class="hero"><h1>💡 Hint Panel</h1><p>Graduated hints — from vague to explicit</p></div>', unsafe_allow_html=True)

    if not st.session_state['quiz_ready'] or not st.session_state['hints_data']:
        st.info("No quiz loaded. Go to **Article Input** and submit first.")
        if st.button("← Go to Article Input"):
            st.session_state['page'] = "📝 Article Input"
            st.rerun()
        st.stop()

    st.markdown(f"**Question:** {st.session_state['question']}")
    st.divider()

    hints = st.session_state['hints_data']
    revealed = st.session_state['hints_revealed']
    labels_h = ['A', 'B', 'C', 'D']
    correct_idx = st.session_state['correct_idx']

    icons = ['🔍', '🔎', '🎯']
    colors = ['#6c63ff', '#ffc107', '#00d68f']

    for i, hint in enumerate(hints):
        if i < revealed:
            st.markdown(f"""
            <div class="hint-box" style="border-left-color: {colors[min(i,2)]};">
                <div class="hint-label">{icons[min(i,2)]} Hint {i+1} — {hint.get('level', '')}</div>
                {hint['text']}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="hint-box" style="opacity: 0.4; border-left-color: #555;">
                <div class="hint-label">🔒 Hint {i+1} — Locked</div>
                Click "Reveal Next Hint" to unlock.
            </div>
            """, unsafe_allow_html=True)

    st.markdown("")
    col1, col2 = st.columns(2)
    with col1:
        if revealed < len(hints):
            if st.button(f"🔓 Reveal Hint {revealed + 1}", use_container_width=True, type="primary"):
                st.session_state['hints_revealed'] = revealed + 1
                st.rerun()
        else:
            st.success("All hints revealed!")
    with col2:
        if revealed >= len(hints):
            if st.button("🎯 Reveal Answer", use_container_width=True, type="primary"):
                if correct_idx is not None:
                    st.balloons()
                    st.success(f"The correct answer is **{labels_h[correct_idx]}) {st.session_state['choices'][correct_idx]}**")
                else:
                    st.warning("No correct answer recorded for this question.")
        else:
            st.caption("Use all hints first to unlock the answer.")

    st.divider()
    if st.button("← Back to Quiz", use_container_width=True):
        st.session_state['page'] = "❓ Quiz View"
        st.rerun()


# ═════════════════════════════════════════════════════
# PAGE 4 — ANALYTICS DASHBOARD
# ═════════════════════════════════════════════════════
elif page == "📊 Dashboard":
    st.markdown('<div class="hero"><h1>📊 Analytics Dashboard</h1><p>Model performance on your session inferences</p></div>', unsafe_allow_html=True)

    log = st.session_state['inference_log']
    
    if not log:
        st.info("No inferences yet. Complete some quizzes to see analytics here.")
        if st.button("← Go to Article Input"):
            st.session_state['page'] = "📝 Article Input"
            st.rerun()
        st.stop()

    metrics = comp.compute_session_metrics(log)
   
    # Top row: summary cards
    st.markdown("### 📈 Session Overview")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""<div class="metric-card"><div class="value">{len(log)}</div><div class="label">Total Inferences</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card"><div class="value">{metrics['avg_latency_s']:.3f}s</div><div class="label">Avg Latency</div></div>""", unsafe_allow_html=True)
    with c3:
        best = max(metrics['model_metrics'], key=lambda m: m['accuracy'])
        st.markdown(f"""<div class="metric-card"><div class="value">{best['accuracy']:.1%}</div><div class="label">Best Accuracy ({best['model']})</div></div>""", unsafe_allow_html=True)

    st.divider()

    # Model A performance table
    st.markdown("### 🅰️ Model A — Answer Verification")
    perf_df = pd.DataFrame(metrics['model_metrics'])
    perf_df = perf_df.set_index('model')

    # Bar chart
    fig = go.Figure()
    metric_names = ['accuracy', 'f1', 'precision', 'recall']
    colors_bar = ['#6c63ff', '#00d68f', '#ffc107', '#ff6b6b']
    x_vals = perf_df.index.tolist()

    for i, m in enumerate(metric_names):
        y_vals = [v if (v is not None and not np.isnan(v)) else 0.0 
                for v in perf_df[m].tolist()]  # ← fill NaN with 0
        fig.add_trace(go.Bar(
            name=m.capitalize(),
            x=x_vals,
            y=y_vals,
            marker_color=colors_bar[i],
            text=[f"{v:.2%}" for v in y_vals],
            textposition='outside',   # ← 'auto' causes rotated text when bars are narrow
            textangle=0,              # ← force horizontal label text
        ))

    fig.update_layout(
        barmode='group',
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=380,
        margin=dict(t=50, b=40),     # ← more top margin for outside labels
        legend=dict(orientation='h', y=1.12),
        yaxis=dict(range=[0, 1.25]), # ← raised ceiling so 'outside' labels fit
        bargap=0.2,
        bargroupgap=0.05,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(perf_df.style.format("{:.4f}"), use_container_width=True)

    # Confusion matrix (per model)
    st.markdown("### 🔢 Confusion Matrices")
    label_set = sorted(set([e['y_true'] for e in log]))
    tabs = st.tabs(["LR", "SVM", "K-Means", "Ensemble"])
    pred_keys = ['y_pred_lr', 'y_pred_svm', 'y_pred_km', 'y_pred_ens']
    for tab, pk in zip(tabs, pred_keys):
        with tab:
            y_true = [e['y_true'] for e in log]
            y_pred = [e[pk] for e in log]
            all_labels = sorted(set(y_true + y_pred))
            from sklearn.metrics import confusion_matrix as cm_fn
            cm = cm_fn(y_true, y_pred, labels=all_labels)
            fig_cm = px.imshow(
                cm, x=all_labels, y=all_labels,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                color_continuous_scale=["#1a1d29", "#6c63ff", "#00d68f"],
                text_auto=True,
            )
            fig_cm.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                height=350,
                margin=dict(t=20, b=20),
            )
            st.plotly_chart(fig_cm, use_container_width=True, key=f"cm_{pk}")

    # Latency over time
    st.markdown("### ⏱ Inference Latency")
    lat_df = pd.DataFrame([{'Inference #': i+1, 'Latency (s)': e['latency']} for i, e in enumerate(log)])
    fig_lat = px.line(lat_df, x='Inference #', y='Latency (s)',
                      markers=True, template='plotly_dark')
    fig_lat.update_traces(line_color='#6c63ff', marker_color='#00d68f')
    fig_lat.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=300)
    st.plotly_chart(fig_lat, use_container_width=True)

    # Session log table + export
    st.markdown("### 📋 Session Log")
    log_df = pd.DataFrame(log)
    st.dataframe(log_df, use_container_width=True)

    csv = log_df.to_csv(index=False)
    st.download_button(
        "📥 Export Session to CSV",
        data=csv,
        file_name="session_results.csv",
        mime="text/csv",
        use_container_width=True,
    )
