"""
Model B -- Distractor Generator (Improved)
Key fix: match candidate TYPE to answer TYPE
"""

import os, re, string
import numpy as np
import pandas as pd
import joblib
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from scipy.sparse import csr_matrix
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
# ANSWER TYPE DETECTION
# This is the KEY fix — match distractor type to answer type
# ─────────────────────────────────────────────────────

# Semantic category pools — used when answer matches a category
# These are plausible alternatives of the same TYPE
SEMANTIC_POOLS = {
    'publication': [
        'a newspaper report', 'a travel magazine', 'a history textbook',
        'an agricultural book', 'a science journal', 'a fiction novel',
        'a biography', 'a sports magazine', 'a cooking book',
        'a medical journal', 'a fashion magazine', 'a comic book',
        'a geography textbook', 'a business report', 'a poetry collection'
    ],
    'emotion': [
        'happy', 'sad', 'angry', 'excited', 'nervous', 'proud',
        'disappointed', 'surprised', 'frightened', 'confused',
        'satisfied', 'worried', 'lonely', 'hopeful', 'ashamed'
    ],
    'time': [
        'in the morning', 'in the evening', 'at noon', 'at midnight',
        'last week', 'next month', 'years ago', 'recently',
        'in the afternoon', 'during winter', 'every day', 'once a week'
    ],
    'place': [
        'at school', 'at home', 'in the hospital', 'in the library',
        'at the market', 'in the park', 'at the office', 'in the city',
        'at the airport', 'in the countryside', 'at the station'
    ],
    'person': [
        'the teacher', 'the doctor', 'the student', 'the manager',
        'the author', 'the scientist', 'the engineer', 'the reporter',
        'the parent', 'the professor', 'the policeman', 'the nurse'
    ],
    'reason': [
        'to save money', 'to get exercise', 'to make friends',
        'to learn new skills', 'to help others', 'to pass the exam',
        'to earn more money', 'to improve health', 'to have fun',
        'to solve the problem', 'to finish the project'
    ],
    'adjective': [
        'interesting', 'boring', 'exciting', 'terrible', 'wonderful',
        'difficult', 'easy', 'important', 'dangerous', 'useful',
        'strange', 'beautiful', 'expensive', 'popular', 'successful'
    ],
    'number': [
        'one', 'two', 'three', 'four', 'five', 'ten', 'twenty',
        'fifty', 'a hundred', 'thousands', 'millions', 'a few', 'many'
    ]
}

def detect_answer_category(answer):
    """
    Detect what semantic category the answer belongs to.
    This tells us what KIND of distractors to generate.
    """
    a = answer.lower().strip()

    # Publication types
    pub_words = ['newspaper', 'magazine', 'textbook', 'book', 'journal',
                 'novel', 'report', 'article', 'story', 'diary', 'letter']
    if any(w in a for w in pub_words):
        return 'publication'

    # Emotions / feelings
    emotion_words = ['happy', 'sad', 'angry', 'excited', 'nervous', 'proud',
                     'pleased', 'upset', 'afraid', 'worried', 'surprised',
                     'disappointed', 'frightened', 'delighted', 'confused']
    if any(w in a for w in emotion_words):
        return 'emotion'

    # Time expressions
    time_words = ['morning', 'evening', 'night', 'noon', 'week', 'month',
                  'year', 'day', 'hour', 'ago', 'later', 'recently',
                  'monday', 'tuesday', 'january', 'february']
    if any(w in a for w in time_words):
        return 'time'

    # Place expressions
    place_words = ['school', 'home', 'hospital', 'library', 'market',
                   'park', 'office', 'city', 'country', 'room', 'street']
    if any(w in a for w in place_words):
        return 'place'

    # Person roles
    person_words = ['teacher', 'doctor', 'student', 'manager', 'author',
                    'scientist', 'engineer', 'reporter', 'professor']
    if any(w in a for w in person_words):
        return 'person'

    # Reason phrases
    reason_words = ['to ', 'in order', 'because', 'so that']
    if any(a.startswith(w) or w in a for w in reason_words):
        return 'reason'

    # Single adjective
    adjective_words = ['interesting', 'boring', 'exciting', 'terrible',
                       'easy', 'difficult', 'important', 'dangerous',
                       'wonderful', 'useful', 'strange', 'beautiful']
    if a in adjective_words or (len(a.split()) == 1 and a.isalpha()):
        return 'adjective'

    # Number or quantity
    if any(char.isdigit() for char in a):
        return 'number'

    return 'general'  # fallback


# ─────────────────────────────────────────────────────
# CANDIDATE EXTRACTION — 3 strategies combined
# ─────────────────────────────────────────────────────

def tokenize(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s\'-]', ' ', text)
    return [w for w in text.split() if len(w) > 1]


def extract_ngrams(tokens, n):
    return [' '.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def extract_candidates(passage, correct_answer, question, max_candidates=40):
    """
    3-strategy candidate extraction:

    Strategy 1 — Semantic pool: if answer matches a known category,
                 use pre-built pool of same-type alternatives.

    Strategy 2 — Passage phrases: extract ngrams from passage
                 that match answer length and type.

    Strategy 3 — Answer variants: modify the answer slightly
                 to create near-miss distractors.
    """
    answer_lower = correct_answer.lower().strip()
    answer_words = tokenize(answer_lower)
    answer_len   = len(answer_words)
    category     = detect_answer_category(correct_answer)

    candidates = []

    # ── Strategy 1: Semantic pool ─────────────────────
    if category in SEMANTIC_POOLS:
        pool = SEMANTIC_POOLS[category]
        for p in pool:
            if p.lower() != answer_lower:
                candidates.append(p)

    # ── Strategy 2: Passage phrases ───────────────────
    passage_tokens = tokenize(passage)
    word_freq = Counter(passage_tokens)

    # Extract ngrams of SAME LENGTH as answer (±1 word)
    for n in range(max(1, answer_len - 1), answer_len + 2):
        ngrams = extract_ngrams(passage_tokens, n)
        ngram_freq = Counter(ngrams)
        for ng, freq in ngram_freq.most_common(30):
            ng_words = ng.split()
            # Skip if all stop words
            if all(w in STOP_WORDS for w in ng_words):
                continue
            # Skip if it IS the answer
            if ng == answer_lower:
                continue
            # Skip if overlap > 80% with answer
            ng_set  = set(ng_words)
            ans_set = set(answer_words)
            if len(ng_set) > 0:
                overlap = len(ng_set & ans_set) / len(ng_set)
                if overlap > 0.8:
                    continue
            candidates.append(ng)

    # ── Strategy 3: Answer variants ───────────────────
    # Swap key content words with semantically related terms
    if answer_len == 1:
        # Single word — try common substitutions
        word_subs = {
            'hands': ['feet', 'eyes', 'mouth', 'arms', 'ears'],
            'easy':  ['hard', 'simple', 'boring', 'exciting', 'quick'],
            'big':   ['small', 'large', 'tall', 'wide', 'heavy'],
            'good':  ['bad', 'great', 'poor', 'better', 'worse'],
            'fast':  ['slow', 'quick', 'steady', 'careful', 'smooth'],
        }
        for key, subs in word_subs.items():
            if key in answer_lower:
                candidates.extend(subs)

    # Remove the correct answer and duplicates
    seen = set()
    filtered = []
    for c in candidates:
        c_lower = c.lower().strip()
        if c_lower in seen:
            continue
        if c_lower == answer_lower:
            continue
        if c_lower in answer_lower or answer_lower in c_lower:
            continue
        seen.add(c_lower)
        filtered.append(c)

    return filtered[:max_candidates]


# ─────────────────────────────────────────────────────
# FEATURE ENGINEERING (unchanged — works well)
# ─────────────────────────────────────────────────────

def build_corpus_vectorizer(df, max_features=5000):
    all_text = (
        df['article'].astype(str) + ' ' +
        df['question'].astype(str) + ' ' +
        df['option'].astype(str)
    )
    vec = CountVectorizer(
        max_features=max_features,
        binary=True,
        stop_words='english',
        min_df=3
    )
    vec.fit(all_text)
    return vec


def cosine_sim_sparse(v1, v2):
    dot = v1.multiply(v2).sum()
    n1  = np.sqrt(v1.multiply(v1).sum())
    n2  = np.sqrt(v2.multiply(v2).sum())
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(dot / (n1 * n2))


def char_match_score(candidate, answer):
    c = candidate.lower()
    a = answer.lower()
    if not c or not a:
        return 0.0
    shared = sum(1 for ch in c if ch in a)
    return shared / max(len(c), len(a))


def compute_candidate_features(candidate, correct_answer, passage,
                                question, vectorizer, passage_tokens):
    cand_vec   = vectorizer.transform([candidate])
    answer_vec = vectorizer.transform([correct_answer])
    cos_sim    = cosine_sim_sparse(cand_vec, answer_vec)

    q_vec      = vectorizer.transform([question])
    cos_sim_q  = cosine_sim_sparse(cand_vec, q_vec)

    char_score = char_match_score(candidate, correct_answer)

    cand_words = tokenize(candidate)
    word_freq  = Counter(passage_tokens)
    avg_freq   = np.mean([word_freq.get(w, 0) for w in cand_words]) if cand_words else 0
    max_freq   = max([word_freq.get(w, 0) for w in cand_words], default=0)

    cand_len   = len(cand_words)
    answer_len = len(tokenize(correct_answer))
    len_ratio  = cand_len / (answer_len + 1e-9)

    answer_words = set(tokenize(correct_answer))
    overlap      = len(set(cand_words) & answer_words) / (len(answer_words) + 1e-9)

    verbatim = 1.0 if candidate.lower() in passage.lower() else 0.0

    # NEW: does candidate match answer category?
    cand_category   = detect_answer_category(candidate)
    answer_category = detect_answer_category(correct_answer)
    same_category   = 1.0 if cand_category == answer_category else 0.0

    # NEW: length difference
    len_diff = abs(cand_len - answer_len)

    return [
        cos_sim, cos_sim_q, char_score,
        avg_freq, max_freq,
        cand_len, len_ratio, len_diff,
        overlap, verbatim,
        same_category   # ← new feature
    ]


# ─────────────────────────────────────────────────────
# RANKER TRAINING (unchanged structure)
# ─────────────────────────────────────────────────────

def build_ranker_data(df, vectorizer, sample_size=5000):
    print("Building distractor ranker training data...")

    groups     = df.groupby(['article', 'question']).filter(lambda g: len(g) == 4)
    group_list = list(groups.groupby(['article', 'question']))

    rng = np.random.default_rng(42)
    if len(group_list) > sample_size:
        indices    = rng.choice(len(group_list), size=sample_size, replace=False)
        group_list = [group_list[i] for i in indices]

    X_features = []
    y_labels   = []
    count_pos  = 0
    count_neg  = 0

    for key, grp in group_list:
        passage  = str(grp.iloc[0]['article'])
        question = str(grp.iloc[0]['question'])
        passage_tokens = tokenize(passage)

        correct_row = grp[grp['is_correct'] == 1]
        if len(correct_row) == 0:
            continue
        correct_answer  = str(correct_row.iloc[0]['option'])
        ref_distractors = grp[grp['is_correct'] == 0]['option'].tolist()
        ref_set = {str(d).lower().strip() for d in ref_distractors}

        candidates = extract_candidates(passage, correct_answer, question)

        for cand in candidates:
            feats = compute_candidate_features(
                cand, correct_answer, passage, question,
                vectorizer, passage_tokens
            )
            cand_lower = cand.lower().strip()
            is_ref = 0
            for ref in ref_set:
                if cand_lower == ref or cand_lower in ref or ref in cand_lower:
                    is_ref = 1
                    break
            X_features.append(feats)
            y_labels.append(is_ref)
            if is_ref: count_pos += 1
            else:      count_neg += 1

        # Always add reference distractors as positive examples
        for dist in ref_distractors:
            feats = compute_candidate_features(
                str(dist), correct_answer, passage, question,
                vectorizer, passage_tokens
            )
            X_features.append(feats)
            y_labels.append(1)
            count_pos += 1

    print(f"  Positive (reference): {count_pos}")
    print(f"  Negative (non-distractor): {count_neg}")
    return np.array(X_features), np.array(y_labels)


# ─────────────────────────────────────────────────────
# GENERATE DISTRACTORS
# ─────────────────────────────────────────────────────

def generate_distractors(passage, question, correct_answer,
                         ranker, vectorizer, top_k=3):
    passage_tokens = tokenize(passage)
    candidates = extract_candidates(passage, correct_answer,
                                    question, max_candidates=40)

    if not candidates:
        return ["(no distractor found)"] * top_k

    feature_matrix = np.array([
        compute_candidate_features(
            c, correct_answer, passage, question,
            vectorizer, passage_tokens
        )
        for c in candidates
    ])

    scores = ranker.predict_proba(feature_matrix)[:, 1]

    # Diversity-aware selection
    selected   = []
    used_words = set()
    for idx in np.argsort(scores)[::-1]:
        cand       = candidates[idx]
        cand_words = set(tokenize(cand))
        if used_words and len(cand_words & used_words) / (len(cand_words) + 1e-9) > 0.5:
            continue
        selected.append(cand)
        used_words.update(cand_words)
        if len(selected) >= top_k:
            break

    while len(selected) < top_k:
        selected.append("(no distractor found)")

    return selected


# ─────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────

def evaluate_distractors(df, ranker, vectorizer, n_eval=500):
    """
    Evaluate distractor generation with:
      - Precision / Recall / F1 on recovering reference distractors
      - Distractor ranker accuracy (top candidate != correct answer)
      - Confusion matrix
    """
    print(f"\nEvaluating on {n_eval} question groups...")

    groups = df.groupby(['article', 'question']).filter(lambda g: len(g) == 4)
    group_list = list(groups.groupby(['article', 'question']))

    rng = np.random.default_rng(99)
    if len(group_list) > n_eval:
        indices = rng.choice(len(group_list), size=n_eval, replace=False)
        group_list = [group_list[i] for i in indices]

    total_ref = 0          # total reference distractors
    total_gen = 0          # total generated distractors
    total_match = 0        # generated that match a reference
    top1_not_answer = 0    # top-1 candidate is NOT the correct answer
    total_questions = 0

    y_true_all = []
    y_pred_all = []

    for key, grp in group_list:
        passage  = str(grp.iloc[0]['article'])
        question = str(grp.iloc[0]['question'])

        correct_row = grp[grp['is_correct'] == 1]
        if len(correct_row) == 0:
            continue
        correct_answer = str(correct_row.iloc[0]['option'])
        ref_distractors = [str(d).lower().strip()
                           for d in grp[grp['is_correct'] == 0]['option'].tolist()]

        # Generate distractors
        gen_distractors = generate_distractors(
            passage, question, correct_answer, ranker, vectorizer, top_k=3
        )

        total_questions += 1

        # Check top-1 is not the answer (ranker accuracy)
        if gen_distractors[0].lower().strip() != correct_answer.lower().strip():
            top1_not_answer += 1

        # Match generated vs reference
        gen_set = {d.lower().strip() for d in gen_distractors
                   if d != "(no distractor found)"}
        ref_set = set(ref_distractors)

        # Count matches (partial match: generated is substring of ref or vice versa)
        matches = 0
        for g in gen_set:
            for r in ref_set:
                if g in r or r in g:
                    matches += 1
                    break

        total_gen   += len(gen_set)
        total_ref   += len(ref_set)
        total_match += matches

        # For per-distractor classification report
        for r in ref_set:
            matched = any(g in r or r in g for g in gen_set)
            y_true_all.append(1)
            y_pred_all.append(1 if matched else 0)

    # Compute metrics
    precision = total_match / (total_gen + 1e-9)
    recall    = total_match / (total_ref + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)
    ranker_acc = top1_not_answer / (total_questions + 1e-9)

    print("\n" + "=" * 55)
    print("  MODEL B: DISTRACTOR GENERATION RESULTS")
    print("=" * 55)
    print(f"  Questions evaluated   : {total_questions}")
    print(f"  Total generated       : {total_gen}")
    print(f"  Total reference       : {total_ref}")
    print(f"  Matches found         : {total_match}")
    print(f"\n  Precision             : {precision:.4f}")
    print(f"  Recall                : {recall:.4f}")
    print(f"  F1 Score              : {f1:.4f}")
    print(f"  Ranker Accuracy       : {ranker_acc:.4f}")
    print(f"    (top-1 distractor is not the correct answer)")

    # Confusion matrix for reference distractor recovery
    if y_true_all and y_pred_all:
        cm = confusion_matrix(y_true_all, y_pred_all, labels=[0, 1])
        print(f"\n  Confusion Matrix (reference distractor recovery):")
        if cm.shape == (2, 2):
            print(f"    TN: {cm[0][0]:<5} | FP: {cm[0][1]:<5}")
            print(f"    FN: {cm[1][0]:<5} | TP: {cm[1][1]:<5}")
        else:
            print(f"    {cm}")

    print("=" * 55)
    return precision, recall, f1


# ─────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────

if __name__ == "__main__":

    # 1. Load data
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    for col in ['article', 'question', 'option']:
        df[col] = df[col].fillna('').astype(str)

    # 2. Build One-Hot corpus vectorizer
    print("Building One-Hot Encoding vocabulary...")
    vectorizer = build_corpus_vectorizer(df, max_features=5000)
    print(f"  Vocabulary size: {len(vectorizer.vocabulary_)}")

    # 3. Build training data for ranker
    X_ranker, y_ranker = build_ranker_data(df, vectorizer, sample_size=5000)
    print(f"  Feature matrix shape: {X_ranker.shape}")
    print(f"  Label distribution: 1s={y_ranker.sum()}, 0s={len(y_ranker)-y_ranker.sum()}")

    # 4. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_ranker, y_ranker, test_size=0.2, random_state=42, stratify=y_ranker
    )

    # 5. Train Logistic Regression ranker
    print("\nTraining Logistic Regression distractor ranker...")
    lr_ranker = LogisticRegression(
        class_weight='balanced', C=1.0, max_iter=1000, solver='lbfgs'
    )
    lr_ranker.fit(X_train, y_train)

    y_pred_lr = lr_ranker.predict(X_test)
    print("\n-- Logistic Regression Ranker Results --")
    print(classification_report(y_test, y_pred_lr,
                                target_names=["Non-Distractor", "Distractor"]))

    # 6. Also train Random Forest for comparison
    print("Training Random Forest distractor ranker...")
    rf_ranker = RandomForestClassifier(
        n_estimators=100, class_weight='balanced',
        random_state=42, n_jobs=-1
    )
    rf_ranker.fit(X_train, y_train)

    y_pred_rf = rf_ranker.predict(X_test)
    print("\n-- Random Forest Ranker Results --")
    print(classification_report(y_test, y_pred_rf,
                                target_names=["Non-Distractor", "Distractor"]))

    # 7. Pick best ranker
    f1_lr = f1_score(y_test, y_pred_lr, average='macro')
    f1_rf = f1_score(y_test, y_pred_rf, average='macro')
    best_ranker = lr_ranker if f1_lr >= f1_rf else rf_ranker
    best_name   = "Logistic Regression" if f1_lr >= f1_rf else "Random Forest"
    print(f"\nBest ranker: {best_name} (F1={max(f1_lr, f1_rf):.4f})")

    # 8. Save models
    joblib.dump(best_ranker,  os.path.join(MODEL_DIR, 'distractor_ranker.pkl'))
    joblib.dump(vectorizer,   os.path.join(MODEL_DIR, 'distractor_vectorizer.pkl'))
    print(f"Models saved to {MODEL_DIR}/")

    # 9. Demo: generate distractors for sample questions
    print("\n" + "=" * 55)
    print("  DEMO: Generated Distractors")
    print("=" * 55)

    demo_groups = df.groupby(['article', 'question']).filter(lambda g: len(g) == 4)
    demo_list   = list(demo_groups.groupby(['article', 'question']))

    rng = np.random.default_rng(123)
    demo_indices = rng.choice(len(demo_list), size=min(5, len(demo_list)), replace=False)

    for i in demo_indices:
        key, grp = demo_list[i]
        passage  = str(grp.iloc[0]['article'])
        question = str(grp.iloc[0]['question'])
        correct  = str(grp[grp['is_correct'] == 1].iloc[0]['option'])
        ref_dist = grp[grp['is_correct'] == 0]['option'].tolist()

        gen_dist = generate_distractors(
            passage, question, correct, best_ranker, vectorizer, top_k=3
        )

        print(f"\n  Q: {question[:80]}...")
        print(f"  Correct Answer: {correct}")
        print(f"  Reference Distractors:")
        for d in ref_dist:
            print(f"    - {d}")
        print(f"  Generated Distractors:")
        for d in gen_dist:
            print(f"    * {d}")
        print("-" * 55)

    # 10. Full evaluation
    evaluate_distractors(df, best_ranker, vectorizer, n_eval=500)
