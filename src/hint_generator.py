"""
Model B -- Extractive Hint Generation with ML Scoring

Two strategies (per project spec 5.3.2):

  1. Extractive: score each sentence by relevance to the question
     using cosine similarity (One-Hot / BoW representations),
     then surface the top-K sentences as ranked hints.

  2. ML-scored: train a Logistic Regression on sentence features
     (keyword overlap, position in passage, sentence length)
     to score and rank sentences as hints.

Evaluation:
  - Precision @ K: fraction of top-K hint sentences that overlap
    with the gold key sentence (the sentence containing the answer).
  - R2 Score: how well predicted relevance scores correlate with
    true relevance labels.
"""

import os
import re
import numpy as np
import pandas as pd
import joblib
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, r2_score
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = 'data/transformed/train.csv'
MODEL_DIR = 'models/model_b'
os.makedirs(MODEL_DIR, exist_ok=True)

STOP_WORDS = {
    'the','a','an','is','was','are','were','be','been','being',
    'have','has','had','do','does','did','will','would','shall',
    'should','may','might','can','could','must','it','its','in',
    'on','at','to','for','of','and','or','but','not','no','by',
    'with','from','as','this','that','these','those','he','she',
    'they','we','you','i','me','my','his','her','our','your',
    'their','them','us','what','which','who','whom','how','when',
    'where','why','if','so','very','too','also','just','then',
    'than','more','most','some','any','all','each','every','much',
    'many','own','other','about','up','out','into','over','after',
    'before','between','under','again','there','here','once','said'
}


# ─────────────────────────────────────────────────────
# SENTENCE SPLITTING
# ─────────────────────────────────────────────────────

def split_sentences(text):
    """Split passage into sentences."""
    text = str(text).strip()
    text = re.sub(r'\s+', ' ', text)
    # Split on sentence-ending punctuation followed by space or end
    sentences = re.split(r'(?<=[.!?])\s+', text)
    # Filter out very short fragments
    sentences = [s.strip() for s in sentences if len(s.split()) >= 4]
    return sentences


def tokenize(text):
    """Simple tokenizer for feature computation."""
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return [w for w in text.split() if len(w) > 1]


def content_words(text):
    """Extract content words (non-stop words) from text."""
    return {w for w in tokenize(text) if w not in STOP_WORDS and len(w) > 2}


# ─────────────────────────────────────────────────────
# STRATEGY 1: Extractive (Cosine Similarity)
# ─────────────────────────────────────────────────────

def cosine_sim_bow(text1, text2, vectorizer):
    """Cosine similarity between two texts using BoW vectors."""
    v1 = vectorizer.transform([text1])
    v2 = vectorizer.transform([text2])
    dot = v1.multiply(v2).sum()
    n1  = np.sqrt(v1.multiply(v1).sum())
    n2  = np.sqrt(v2.multiply(v2).sum())
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(dot / (n1 * n2))


def extractive_hints(passage, question, vectorizer, top_k=3):
    """
    Strategy 1: Score each sentence by cosine similarity to
    the question, return top-K as ranked hints.
    """
    sentences = split_sentences(passage)
    if not sentences:
        return []

    scored = []
    for sent in sentences:
        sim = cosine_sim_bow(sent, question, vectorizer)
        scored.append((sent, sim))

    # Sort by similarity (highest first)
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


# ─────────────────────────────────────────────────────
# STRATEGY 2: ML-Scored (Logistic Regression)
# ─────────────────────────────────────────────────────

def compute_sentence_features(sentence, question, answer, passage,
                               sent_idx, total_sents):
    """
    Compute handcrafted features for a sentence (for hint scoring).

    Features:
      1. Keyword overlap with question (content words)
      2. Keyword overlap with answer
      3. Position in passage (normalized 0-1)
      4. Sentence length (word count, normalized)
      5. Is it the first sentence?
      6. Is it the last sentence?
      7. Content word density
      8. Answer words present in sentence (ratio)
      9. Question words present in sentence (ratio)
    """
    sent_words = set(tokenize(sentence))
    q_content  = content_words(question)
    a_content  = content_words(answer)
    s_content  = content_words(sentence)

    # 1. Keyword overlap with question
    q_overlap = len(s_content & q_content) / (len(q_content) + 1e-9)

    # 2. Keyword overlap with answer
    a_overlap = len(s_content & a_content) / (len(a_content) + 1e-9)

    # 3. Position in passage (0 = first, 1 = last)
    position = sent_idx / (total_sents + 1e-9)

    # 4. Sentence length (normalized by 30 words)
    sent_len = len(sentence.split()) / 30.0

    # 5. Is first sentence?
    is_first = 1.0 if sent_idx == 0 else 0.0

    # 6. Is last sentence?
    is_last = 1.0 if sent_idx == total_sents - 1 else 0.0

    # 7. Content word density
    all_words = tokenize(sentence)
    density = len(s_content) / (len(all_words) + 1e-9)

    # 8. Answer words present (ratio)
    a_words = set(tokenize(answer))
    a_in_sent = len(sent_words & a_words) / (len(a_words) + 1e-9)

    # 9. Question words present (ratio)
    q_words = set(tokenize(question))
    q_in_sent = len(sent_words & q_words) / (len(q_words) + 1e-9)

    return [
        q_overlap,    # keyword overlap with question
        a_overlap,    # keyword overlap with answer
        position,     # position in passage
        sent_len,     # sentence length
        is_first,     # first sentence flag
        is_last,      # last sentence flag
        density,      # content word density
        a_in_sent,    # answer word coverage
        q_in_sent,    # question word coverage
    ]


# ─────────────────────────────────────────────────────
# BUILD TRAINING DATA
# ─────────────────────────────────────────────────────

def find_gold_sentence(passage, answer):
    """
    Find the sentence in the passage that best contains the answer.
    Returns the index of the best-matching sentence.
    """
    sentences = split_sentences(passage)
    if not sentences:
        return -1

    answer_words = content_words(answer)
    if not answer_words:
        return 0

    best_idx   = 0
    best_score = -1

    for i, sent in enumerate(sentences):
        sent_words = content_words(sent)
        overlap = len(sent_words & answer_words) / (len(answer_words) + 1e-9)
        # Also check verbatim
        if answer.lower().strip() in sent.lower():
            overlap += 1.0
        if overlap > best_score:
            best_score = overlap
            best_idx = i

    return best_idx


def build_hint_training_data(df, sample_size=5000):
    """
    Build labelled data for the hint sentence scorer.

    Label = 1 if the sentence contains (or best overlaps) the answer.
    Label = 0 otherwise.
    Also compute a continuous relevance score for R2 evaluation.
    """
    print("Building hint scorer training data...")

    groups = df.groupby(['article', 'question']).filter(lambda g: len(g) == 4)
    group_list = list(groups.groupby(['article', 'question']))

    rng = np.random.default_rng(42)
    if len(group_list) > sample_size:
        indices = rng.choice(len(group_list), size=sample_size, replace=False)
        group_list = [group_list[i] for i in indices]

    X_features  = []
    y_binary    = []  # 1 = gold hint sentence, 0 = other
    y_relevance = []  # continuous relevance score

    count_pos = 0
    count_neg = 0

    for key, grp in group_list:
        passage  = str(grp.iloc[0]['article'])
        question = str(grp.iloc[0]['question'])

        correct_row = grp[grp['is_correct'] == 1]
        if len(correct_row) == 0:
            continue
        answer = str(correct_row.iloc[0]['option'])

        sentences = split_sentences(passage)
        if len(sentences) < 2:
            continue

        # Find gold sentence (the one containing the answer)
        gold_idx = find_gold_sentence(passage, answer)
        answer_words = content_words(answer)

        for i, sent in enumerate(sentences):
            feats = compute_sentence_features(
                sent, question, answer, passage, i, len(sentences)
            )
            X_features.append(feats)

            # Binary label: is this the gold hint sentence?
            is_gold = 1 if i == gold_idx else 0
            y_binary.append(is_gold)

            # Continuous relevance: overlap ratio with answer
            s_content = content_words(sent)
            relevance = len(s_content & answer_words) / (len(answer_words) + 1e-9)
            y_relevance.append(relevance)

            if is_gold:
                count_pos += 1
            else:
                count_neg += 1

    print(f"  Gold hint sentences: {count_pos}")
    print(f"  Other sentences: {count_neg}")

    return (np.array(X_features),
            np.array(y_binary),
            np.array(y_relevance))


# ─────────────────────────────────────────────────────
# HINT GENERATION PIPELINE
# ─────────────────────────────────────────────────────

def generate_hints_extractive(passage, question, vectorizer, top_k=3):
    """Strategy 1: Extractive hints using cosine similarity."""
    return extractive_hints(passage, question, vectorizer, top_k)


def generate_hints_ml(passage, question, answer, scorer, top_k=3):
    """
    Strategy 2: ML-scored hints using trained Logistic Regression.
    Returns top-K sentences ranked by predicted relevance.
    """
    sentences = split_sentences(passage)
    if not sentences:
        return []

    features = np.array([
        compute_sentence_features(
            sent, question, answer, passage, i, len(sentences)
        )
        for i, sent in enumerate(sentences)
    ])

    # Get probability of being a gold hint sentence
    scores = scorer.predict_proba(features)[:, 1]

    # Rank by score
    ranked = sorted(zip(sentences, scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


# ─────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────

def evaluate_hints(df, scorer, vectorizer, n_eval=500):
    """
    Evaluate hint generation with:
      - Precision @ K for both strategies
      - R2 score for ML scorer
    """
    print(f"\nEvaluating hint generation on {n_eval} questions...")

    groups = df.groupby(['article', 'question']).filter(lambda g: len(g) == 4)
    group_list = list(groups.groupby(['article', 'question']))

    rng = np.random.default_rng(99)
    if len(group_list) > n_eval:
        indices = rng.choice(len(group_list), size=n_eval, replace=False)
        group_list = [group_list[i] for i in indices]

    # Metrics accumulators
    ext_hits_at1 = 0
    ext_hits_at3 = 0
    ml_hits_at1  = 0
    ml_hits_at3  = 0
    total        = 0

    y_true_relevance = []
    y_pred_relevance = []

    for key, grp in group_list:
        passage  = str(grp.iloc[0]['article'])
        question = str(grp.iloc[0]['question'])

        correct_row = grp[grp['is_correct'] == 1]
        if len(correct_row) == 0:
            continue
        answer = str(correct_row.iloc[0]['option'])

        sentences = split_sentences(passage)
        if len(sentences) < 2:
            continue

        gold_idx = find_gold_sentence(passage, answer)
        gold_sent = sentences[gold_idx] if 0 <= gold_idx < len(sentences) else ""
        gold_words = content_words(gold_sent)

        total += 1

        # --- Extractive hints ---
        ext_hints = generate_hints_extractive(passage, question, vectorizer, top_k=3)
        for rank, (hint_sent, score) in enumerate(ext_hints):
            hint_words = content_words(hint_sent)
            overlap = len(hint_words & gold_words) / (len(gold_words) + 1e-9)
            if overlap > 0.5:
                if rank == 0:
                    ext_hits_at1 += 1
                ext_hits_at3 += 1
                break

        # --- ML-scored hints ---
        ml_hints = generate_hints_ml(passage, question, answer, scorer, top_k=3)
        for rank, (hint_sent, score) in enumerate(ml_hints):
            hint_words = content_words(hint_sent)
            overlap = len(hint_words & gold_words) / (len(gold_words) + 1e-9)
            if overlap > 0.5:
                if rank == 0:
                    ml_hits_at1 += 1
                ml_hits_at3 += 1
                break

        # --- R2 data: score all sentences ---
        if sentences:
            features = np.array([
                compute_sentence_features(
                    s, question, answer, passage, i, len(sentences)
                )
                for i, s in enumerate(sentences)
            ])
            pred_scores = scorer.predict_proba(features)[:, 1]

            answer_words = content_words(answer)
            for i, s in enumerate(sentences):
                s_content = content_words(s)
                true_rel = len(s_content & answer_words) / (len(answer_words) + 1e-9)
                y_true_relevance.append(true_rel)
                y_pred_relevance.append(pred_scores[i])

    # --- Report ---
    print("\n" + "=" * 55)
    print("  HINT GENERATION EVALUATION")
    print("=" * 55)
    print(f"  Questions evaluated: {total}")

    print(f"\n  -- Extractive (Cosine Similarity) --")
    print(f"  Precision @ 1: {ext_hits_at1 / (total + 1e-9):.4f}")
    print(f"  Precision @ 3: {ext_hits_at3 / (total + 1e-9):.4f}")

    print(f"\n  -- ML-Scored (Logistic Regression) --")
    print(f"  Precision @ 1: {ml_hits_at1 / (total + 1e-9):.4f}")
    print(f"  Precision @ 3: {ml_hits_at3 / (total + 1e-9):.4f}")

    if y_true_relevance and y_pred_relevance:
        r2 = r2_score(y_true_relevance, y_pred_relevance)
        print(f"\n  R2 Score (ML relevance prediction): {r2:.4f}")

    print("=" * 55)

    return {
        'ext_p1': ext_hits_at1 / (total + 1e-9),
        'ext_p3': ext_hits_at3 / (total + 1e-9),
        'ml_p1':  ml_hits_at1  / (total + 1e-9),
        'ml_p3':  ml_hits_at3  / (total + 1e-9),
    }


# ─────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────

if __name__ == "__main__":

    # 1. Load data
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    for col in ['article', 'question', 'option']:
        df[col] = df[col].fillna('').astype(str)

    # 2. Build corpus vectorizer (for extractive cosine similarity)
    print("Building One-Hot Encoding vocabulary...")
    all_text = df['article'].astype(str) + ' ' + df['question'].astype(str)
    vectorizer = CountVectorizer(
        max_features=5000, binary=True,
        stop_words='english', min_df=3
    )
    vectorizer.fit(all_text)
    print(f"  Vocabulary size: {len(vectorizer.vocabulary_)}")

    # 3. Build training data for ML hint scorer
    X_hint, y_binary, y_relevance = build_hint_training_data(
        df, sample_size=5000
    )
    print(f"  Feature matrix shape: {X_hint.shape}")
    print(f"  Label distribution: gold={y_binary.sum()}, "
          f"other={len(y_binary) - y_binary.sum()}")

    # 4. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_hint, y_binary, test_size=0.2, random_state=42, stratify=y_binary
    )

    # 5. Train Logistic Regression hint scorer
    print("\nTraining Logistic Regression hint scorer...")
    hint_scorer = LogisticRegression(
        class_weight='balanced', C=1.0, max_iter=1000, solver='lbfgs'
    )
    hint_scorer.fit(X_train, y_train)

    y_pred = hint_scorer.predict(X_test)
    print("\n-- Hint Scorer Classification Report --")
    print(classification_report(y_test, y_pred,
                                target_names=["Non-Hint", "Hint"]))

    # R2 on test set
    y_test_proba = hint_scorer.predict_proba(X_test)[:, 1]
    # Build continuous relevance for test set
    _, _, y_rel_full = build_hint_training_data(df, sample_size=5000)
    _, y_rel_test = train_test_split(
        y_rel_full, test_size=0.2, random_state=42, stratify=y_binary
    )
    r2 = r2_score(y_rel_test, y_test_proba)
    print(f"  R2 Score (test set): {r2:.4f}")

    # 6. Save models
    joblib.dump(hint_scorer,  os.path.join(MODEL_DIR, 'hint_scorer.pkl'))
    joblib.dump(vectorizer,   os.path.join(MODEL_DIR, 'hint_vectorizer.pkl'))
    print(f"\nModels saved to {MODEL_DIR}/")

    # 7. Demo: show hints for sample questions
    print("\n" + "=" * 55)
    print("  DEMO: Generated Hints")
    print("=" * 55)

    demo_groups = df.groupby(['article', 'question']).filter(
        lambda g: len(g) == 4
    )
    demo_list = list(demo_groups.groupby(['article', 'question']))

    rng = np.random.default_rng(77)
    demo_idx = rng.choice(len(demo_list), size=min(5, len(demo_list)),
                          replace=False)

    for i in demo_idx:
        key, grp = demo_list[i]
        passage  = str(grp.iloc[0]['article'])
        question = str(grp.iloc[0]['question'])
        correct  = str(grp[grp['is_correct'] == 1].iloc[0]['option'])

        print(f"\n  Q: {question[:80]}")
        print(f"  Answer: {correct}")

        # Extractive hints
        ext_hints = generate_hints_extractive(passage, question, vectorizer)
        print(f"\n  Extractive Hints (cosine sim):")
        for j, (sent, score) in enumerate(ext_hints):
            # Truncate long sentences for display
            display = sent[:100] + "..." if len(sent) > 100 else sent
            print(f"    Hint {j+1} (sim={score:.3f}): {display}")

        # ML-scored hints
        ml_hints = generate_hints_ml(passage, question, correct, hint_scorer)
        print(f"\n  ML-Scored Hints (LR):")
        for j, (sent, score) in enumerate(ml_hints):
            display = sent[:100] + "..." if len(sent) > 100 else sent
            print(f"    Hint {j+1} (score={score:.3f}): {display}")

        print("-" * 55)

    # 8. Full evaluation
    evaluate_hints(df, hint_scorer, vectorizer, n_eval=500)
