"""
components.py — Backend API for the Streamlit UI.

Loads all trained models once at import time, then exposes
clean functions the UI can call without touching sklearn internals.
"""

import os
import sys
import time
import random
import numpy as np
import pandas as pd
import joblib
from scipy.sparse import hstack
from collections import Counter
import re
import warnings
warnings.filterwarnings('ignore')
import streamlit as st

# ─────────────────────────────────────────────────────
# PATH SETUP  (works whether you run from project root
#              or from inside the UI/ folder)
# ─────────────────────────────────────────────────────
_this_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_this_dir, '..'))

# Make sure src/ is importable
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

MODEL_A_DIR = os.path.join(PROJECT_ROOT, 'models', 'model_a', 'traditional')
MODEL_B_DIR = os.path.join(PROJECT_ROOT, 'models', 'model_b')
DATA_PATH   = os.path.join(PROJECT_ROOT, 'data', 'transformed', 'train.csv')


# ─────────────────────────────────────────────────────
# LOAD MODELS (done once when this module is imported)
# ─────────────────────────────────────────────────────

def _load_models():
    """Load all pkl artefacts into a dict."""
    m = {}

    # Model A — answer verification
    m['lr']         = joblib.load(os.path.join(MODEL_A_DIR, 'logistic_regression.pkl'))
    m['svm']        = joblib.load(os.path.join(MODEL_A_DIR, 'svm.pkl'))
    m['vectorizer'] = joblib.load(os.path.join(MODEL_A_DIR, 'vectorizer.pkl'))
    m['kmeans']     = joblib.load(os.path.join(MODEL_A_DIR, 'kmeans.pkl'))
    m['cluster_map'] = joblib.load(os.path.join(MODEL_A_DIR, 'cluster_map.pkl'))
    m['question_ranker'] = joblib.load(os.path.join(MODEL_A_DIR, 'question_ranker.pkl'))

    # Model B — distractor generation
    m['dist_ranker']     = joblib.load(os.path.join(MODEL_B_DIR, 'distractor_ranker.pkl'))
    m['dist_vectorizer'] = joblib.load(os.path.join(MODEL_B_DIR, 'distractor_vectorizer.pkl'))

    # Hint generation
    m['hint_scorer']     = joblib.load(os.path.join(MODEL_B_DIR, 'hint_scorer.pkl'))
    m['hint_vectorizer'] = joblib.load(os.path.join(MODEL_B_DIR, 'hint_vectorizer.pkl'))

    return m


MODELS = _load_models()


# ─────────────────────────────────────────────────────
# HANDCRAFTED FEATURES  (same as training)
# ─────────────────────────────────────────────────────

def _compute_handcrafted(df):
    features = []
    for _, row in df.iterrows():
        art_w = set(str(row['article']).lower().split())
        q_w   = set(str(row['question']).lower().split())
        o_w   = set(str(row['option']).lower().split())

        features.append([
            len(o_w & art_w) / (len(o_w) + 1e-9),
            len(q_w & art_w) / (len(q_w) + 1e-9),
            len(o_w & q_w)   / (len(q_w) + 1e-9),
            len(str(row['option']).split()) / 20.0,
            1.0 if str(row['option']).lower() in str(row['article']).lower() else 0.0,
        ])
    return np.array(features)


def _build_features(rows):
    vec = MODELS['vectorizer']
    X_bow  = vec.transform(rows['option'].fillna('').astype(str))
    X_hand = _compute_handcrafted(rows)
    return hstack([X_bow, X_hand]).tocsr()


# ─────────────────────────────────────────────────────
# MODEL A: PREDICT CORRECT ANSWER
# ─────────────────────────────────────────────────────

def predict_answer(article: str, question: str, choices: list[str]) -> dict:
    """
    Run all Model-A classifiers on the 4 choices.
    Returns dict  {model_name: {label, text, scores, score_type}}
    """
    labels = ['A', 'B', 'C', 'D'][:len(choices)]
    rows = pd.DataFrame([{
        'article': article, 'question': question, 'option': str(c)
    } for c in choices])
    for col in ['article', 'question', 'option']:
        rows[col] = rows[col].fillna('').astype(str)
    X = _build_features(rows)

    results = {}

    # LR
    lr_scores = MODELS['lr'].predict_proba(X)[:, 1]
    best = int(np.argmax(lr_scores))
    results['Logistic Regression'] = {
        'label': labels[best], 'text': choices[best],
        'score_type': 'Confidence (higher = better)',
        'scores': {labels[i]: round(float(lr_scores[i]), 4) for i in range(len(choices))}
    }

    # SVM
    svm_scores = MODELS['svm'].decision_function(X)
    best = int(np.argmax(svm_scores))
    results['SVM'] = {
        'label': labels[best], 'text': choices[best],
        'score_type': 'Decision score (higher = better)',
        'scores': {labels[i]: round(float(svm_scores[i]), 4) for i in range(len(choices))}
    }

    # K-Means
    cc = MODELS['cluster_map']['correct_cluster']
    dists = MODELS['kmeans'].transform(X)[:, cc]
    best = int(np.argmin(dists))
    results['K-Means'] = {
        'label': labels[best], 'text': choices[best],
        'score_type': 'Distance to correct centroid (lower = better)',
        'scores': {labels[i]: round(float(dists[i]), 4) for i in range(len(choices))}
    }

    # Ensemble
    votes = [
        int(np.argmax(lr_scores)),
        int(np.argmax(svm_scores)),
        int(np.argmin(dists)),
    ]
    vc = np.bincount(votes, minlength=len(choices))
    best = int(np.argmax(vc))
    results['Ensemble (Majority Vote)'] = {
        'label': labels[best], 'text': choices[best],
        'score_type': 'Votes (higher = more agreement)',
        'scores': {labels[i]: int(vc[i]) for i in range(len(choices))}
    }

    return results


# ─────────────────────────────────────────────────────
# MODEL B: GENERATE DISTRACTORS
# ─────────────────────────────────────────────────────
# Import the heavy functions from source
from model_b_train import (
    generate_distractors as _gen_dist_raw,
    extract_candidates,
    detect_answer_category,
)

def generate_distractors(passage: str, question: str,
                         correct_answer: str, top_k: int = 3) -> list[str]:
    """Generate `top_k` plausible distractors for the MCQ."""
    return _gen_dist_raw(
        passage, question, correct_answer,
        MODELS['dist_ranker'], MODELS['dist_vectorizer'],
        top_k=top_k,
    )


# ─────────────────────────────────────────────────────
# QUESTION GENERATION
# ─────────────────────────────────────────────────────
from model_a_train_generation import generate_question as _gen_q_raw

def generate_question(passage: str, correct_answer: str) -> str:
    return _gen_q_raw(passage, correct_answer, MODELS['question_ranker'])


# ─────────────────────────────────────────────────────
# HINT GENERATION
# ─────────────────────────────────────────────────────
from hint_generator import (
    generate_hints_extractive as _ext_hints,
    generate_hints_ml as _ml_hints,
    split_sentences,
    content_words,
)

def generate_hints(passage: str, question: str,
                   correct_answer: str, n_hints: int = 3) -> list[dict]:
    """
    Return graduated hints (general → specific).
    Each item: {'text': ..., 'score': ..., 'strategy': ...}
    """
    # ML-scored hints (usually better)
    ml = _ml_hints(passage, question, correct_answer,
                   MODELS['hint_scorer'], top_k=n_hints)
    # Extractive (cosine) hints as fallback
    ext = _ext_hints(passage, question,
                     MODELS['hint_vectorizer'], top_k=n_hints)

    hints = []
    used = set()

    # Interleave: prefer ML first, fill with extractive
    sources = list(ml) + list(ext)
    for sent, score in sources:
        key = sent.strip().lower()
        if key in used:
            continue
        used.add(key)
        hints.append({'text': sent.strip(), 'score': round(float(score), 4)})
        if len(hints) >= n_hints:
            break

    # Sort so hint 1 is the most general (lowest score), hint 3 most specific
    hints.sort(key=lambda h: h['score'])

    # Label them
    labels = ['General Clue', 'More Specific', 'Near-Explicit']
    for i, h in enumerate(hints):
        h['level'] = labels[min(i, len(labels) - 1)]

    return hints


# ─────────────────────────────────────────────────────
# RACE DATASET: LOAD RANDOM SAMPLE
# ─────────────────────────────────────────────────────

_race_df = None

def _get_race_df():
    global _race_df
    if _race_df is None:
        _race_df = pd.read_csv(DATA_PATH)
        for col in ['article', 'question', 'option']:
            _race_df[col] = _race_df[col].fillna('').astype(str)
    return _race_df


@st.cache_data(show_spinner=False)
def _get_valid_race_keys():
    df = _get_race_df()
    counts = df.groupby(['article', 'question']).size()
    return list(counts[counts == 4].index)


def load_random_race_sample() -> dict:
    """
    Load one random RACE question group (article + question + 4 options).
    Returns {article, question, choices: [str×4], correct_idx: int, correct_label: str}
    """
    df = _get_race_df()
    valid_keys = _get_valid_race_keys()
    article, question = random.choice(valid_keys)
    
    grp = df[(df['article'] == article) & (df['question'] == question)].reset_index(drop=True)

    choices = grp['option'].tolist()
    correct_mask = grp['is_correct'] == 1
    correct_idx = int(correct_mask.idxmax()) if correct_mask.any() else 0

    labels = ['A', 'B', 'C', 'D']
    return {
        'article':       grp.iloc[0]['article'],
        'question':      grp.iloc[0]['question'],
        'choices':       choices,
        'correct_idx':   correct_idx,
        'correct_label': labels[correct_idx],
    }


# ─────────────────────────────────────────────────────
# ANALYTICS HELPERS
# ─────────────────────────────────────────────────────

def compute_session_metrics(log: list[dict]) -> dict:
    """
    Given a list of inference log entries, compute aggregate metrics.
    Each entry: {y_true, y_pred_lr, y_pred_svm, y_pred_km, y_pred_ens, latency}
    """
    if not log:
        return {}

    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    def _metrics(y_true, y_pred, name):
        # With only one unique class, macro-averaged metrics are misleading.
        # Fall back to accuracy-only display in that case.
        unique_classes = set(y_true) | set(y_pred)
        multi_class = len(unique_classes) > 1

        return {
            'model': name,
            'accuracy':  round(accuracy_score(y_true, y_pred), 4),
            'f1':        round(f1_score(y_true, y_pred, average='macro', zero_division=0), 4) if multi_class else round(accuracy_score(y_true, y_pred), 4),
            'precision': round(precision_score(y_true, y_pred, average='macro', zero_division=0), 4) if multi_class else round(accuracy_score(y_true, y_pred), 4),
            'recall':    round(recall_score(y_true, y_pred, average='macro', zero_division=0), 4) if multi_class else round(accuracy_score(y_true, y_pred), 4),
        }

    y_true = [e['y_true'] for e in log]
    metrics = []
    for key, name in [('y_pred_lr', 'Logistic Regression'),
                      ('y_pred_svm', 'SVM'),
                      ('y_pred_km', 'K-Means'),
                      ('y_pred_ens', 'Ensemble')]:
        y_pred = [e[key] for e in log]
        metrics.append(_metrics(y_true, y_pred, name))

    avg_latency = round(np.mean([e['latency'] for e in log]), 3)
    return {'model_metrics': metrics, 'avg_latency_s': avg_latency}
