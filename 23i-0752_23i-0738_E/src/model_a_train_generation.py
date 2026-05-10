import re
import string
import numpy as np
import pandas as pd
import joblib
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from scipy.sparse import hstack

# ─────────────────────────────────────────────────────
# STEP 1: Extract Candidate Sentences
# ─────────────────────────────────────────────────────

def split_into_sentences(text):
    """Better sentence splitter."""
    text = str(text).strip()
    
    # Normalize whitespace first
    text = re.sub(r'\s+', ' ', text)
    
    # Split on sentence endings
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Clean and filter
    sentences = [s.strip() for s in sentences if len(s.split()) >= 5]
    
    return sentences

def compute_word_overlap(sentence, answer):
    """Better overlap — ignore stop words and short words."""
    stop_words = {'the', 'a', 'an', 'is', 'was', 'are', 'were',
                  'it', 'its', 'in', 'on', 'at', 'to', 'for',
                  'of', 'and', 'or', 'but', 'he', 'she', 'they',
                  'i', 'we', 'you', 'that', 'this', 'with', 'as',
                  'be', 'by', 'from', 'not', 'what', 'have', 'had'}

    # Also filter out very short words (length < 3)
    sentence_words = {w for w in sentence.lower().split() 
                      if w not in stop_words and len(w) > 2}
    answer_words   = {w for w in answer.lower().split() 
                      if w not in stop_words and len(w) > 2}

    if len(answer_words) == 0:
        return 0.0

    overlap = len(sentence_words & answer_words)
    return overlap / len(answer_words)


def extract_candidate_sentences(passage, correct_answer, top_k=3):
    """
    Step 1: Find top-K sentences from the passage that
    most likely contain the answer.
    Returns list of (sentence, score) sorted by score.
    """
    sentences = split_into_sentences(passage)
    
    scored = []
    for sent in sentences:
        score = compute_word_overlap(sent, correct_answer)
        scored.append((sent, score))
    
    # Sort by overlap score, highest first
    scored.sort(key=lambda x: x[1], reverse=True)
    
    # Return top K candidates (with score > 0)
    candidates = [(s, sc) for s, sc in scored if sc > 0]
    
    # If nothing overlaps, just take the first 3 sentences
    if len(candidates) == 0:
        candidates = [(s, 0.0) for s in sentences[:top_k]]
    
    return candidates[:top_k]


# ─────────────────────────────────────────────────────
# STEP 2: Apply Wh-word Templates
# ─────────────────────────────────────────────────────

def detect_answer_type(answer):
    """Improved answer type detection with more categories."""
    answer_lower = answer.lower().strip()
    words = answer_lower.split()

    # ── Count / Number answers → "How many" ──────────
    count_words = {'once', 'twice', 'three', 'four', 'five', 'six',
                   'seven', 'eight', 'nine', 'ten', 'one', 'two',
                   'several', 'many', 'few', 'hundred', 'thousand',
                   'dozen', 'none', 'both', 'all'}
    if any(w in count_words for w in words):
        return 'How many'
    # Pure digits like "3", "42"
    if answer_lower.strip().isdigit():
        return 'How many'

    # ── Time answers → "When" ────────────────────────
    time_words = {'monday','tuesday','wednesday','thursday','friday',
                  'saturday','sunday','january','february','march',
                  'april','may','june','july','august','september',
                  'october','november','december','year','month',
                  'day','week','hour','morning','evening','night',
                  'today','yesterday','tomorrow','ago','later','time',
                  'century','decade','period','date','season',
                  'after','before','during','meanwhile','soon'}
    if any(w in time_words for w in words):
        return 'When'

    # ── Place answers → "Where" ──────────────────────
    place_words = {'school','city','country','street','road','house',
                   'building','park','hospital','market','store',
                   'office','room','place','location','town','village',
                   'restaurant','station','airport','museum','library',
                   'home','here','there','area','region','beach',
                   'forest','garden','hall','church','farm'}
    if any(w in place_words for w in words):
        return 'Where'

    # ── Reason answers → "Why" ───────────────────────
    reason_words = {'because','since','therefore','due','order',
                    'reason','cause','result','effect','hence',
                    'so that', 'in order'}
    if any(w in reason_words for w in words):
        return 'Why'

    # ── Person answers → "Who" (only true person refs)─
    # Note: 'student', 'teacher' etc. are ROLES, not persons
    # Only use Who for actual names or pronouns
    person_words = {'he','she','they','mr','mrs','dr','professor',
                    'father','mother','author','speaker'}
    if any(w in person_words for w in words):
        return 'Who'
    # Single capitalized word that looks like a name
    if len(words) == 1 and answer.strip()[0].isupper() and answer.strip().isalpha():
        return 'Who'
    # Multi-word with title case → probably a name
    if len(words) >= 2 and all(w[0].isupper() for w in answer.strip().split() if w):
        return 'Who'

    # ── Manner answers → "How" ───────────────────────
    manner_words = {'by','through','using','carefully','quickly',
                    'slowly','happily','sadly','easily','hard'}
    if any(w in manner_words for w in words):
        return 'How'

    return 'What'


def _clean_question_grammar(question):
    """Basic grammar cleanup after template substitution."""
    # Remove double spaces
    question = re.sub(r'\s+', ' ', question).strip()
    # Capitalize first letter
    if question:
        question = question[0].upper() + question[1:]
    # Ensure ends with ?
    question = question.rstrip('.!?,;') + '?'
    return question


def _make_direct_question(sentence, wh_word):
    """Convert a short declarative sentence into a direct question.
    
    Only works on short sentences to avoid garbled output.
    E.g., 'He was invited to be an instructor' -> 'What was he invited to be?'
    """
    sent = sentence.strip().rstrip('.!?')
    words = sent.split()
    
    # Only attempt inversion on short-ish sentences
    if len(words) > 15 or len(words) < 4:
        return None
    
    # Try to find a verb to pivot on (only in first 4 words)
    be_verbs = {'is', 'was', 'are', 'were'}
    aux_verbs = {'can', 'could', 'will', 'would', 'shall', 'should',
                 'has', 'had', 'have', 'did', 'do', 'does'}
    
    for i, w in enumerate(words[:4]):
        wl = w.lower().strip(string.punctuation)
        if wl in be_verbs and 0 < i <= 3:
            subject = ' '.join(words[:i])
            verb = words[i]
            rest = ' '.join(words[i+1:])
            q = f"{wh_word} {verb.lower()} {subject.lower()} {rest}"
            return _clean_question_grammar(q)
        if wl in aux_verbs and 0 < i <= 3:
            subject = ' '.join(words[:i])
            verb = words[i]
            rest = ' '.join(words[i+1:])
            q = f"{wh_word} {verb.lower()} {subject.lower()} {rest}"
            return _clean_question_grammar(q)
    
    return None


def _truncate_around_blank(text, max_words=18):
    """Truncate a long sentence but keep context around the blank."""
    words = text.split()
    if len(words) <= max_words:
        return text
    
    # Find the blank position
    blank_idx = None
    for i, w in enumerate(words):
        if '_____' in w:
            blank_idx = i
            break
    
    if blank_idx is not None:
        # Keep a window around the blank
        start = max(0, blank_idx - max_words // 2)
        end = min(len(words), start + max_words)
        truncated = ' '.join(words[start:end])
        if start > 0:
            truncated = '... ' + truncated
        if end < len(words):
            truncated = truncated + ' ...'
        return truncated
    
    # No blank found, just truncate from the start
    return ' '.join(words[:max_words]) + ' ...'


def apply_template(sentence, answer, wh_word):
    """
    Apply question template -- multiple strategies ordered
    from best to acceptable quality.
    """
    sentence_clean = sentence.strip().rstrip('.!?')
    answer_clean   = answer.strip()

    # ── Strategy 1: Direct verbatim replacement ──────────
    # If the full answer appears in the sentence, blank it out
    pattern = re.compile(re.escape(answer_clean), re.IGNORECASE)
    if pattern.search(sentence_clean):
        question = pattern.sub(' _____ ', sentence_clean, count=1)
        question = _truncate_around_blank(question)
        return _clean_question_grammar(question)

    # ── Strategy 2: Multi-word partial match ─────────────
    # For multi-word answers, try replacing a contiguous chunk
    answer_words = [w for w in answer_clean.split() if len(w) > 2]
    if len(answer_words) >= 2:
        for i in range(len(answer_words) - 1):
            bigram = re.escape(answer_words[i]) + r'\s+' + re.escape(answer_words[i+1])
            bigram_pattern = re.compile(bigram, re.IGNORECASE)
            if bigram_pattern.search(sentence_clean):
                question = bigram_pattern.sub(' _____ ', sentence_clean, count=1)
                question = _truncate_around_blank(question)
                return _clean_question_grammar(question)

    # ── Strategy 3: Sentence inversion (short sentences only) ──
    inverted = _make_direct_question(sentence_clean, wh_word)
    if inverted:
        return inverted

    # ── Strategy 4: Fill-in-the-blank at end ─────────────
    # Truncate sentence and add blank at the end
    words = sentence_clean.split()
    if len(words) > 15:
        short = ' '.join(words[:12])
    else:
        short = sentence_clean
    return _clean_question_grammar(f"{short} _____")


def generate_candidate_questions(sentence, answer):
    """
    Generate multiple question variants for one sentence.
    Returns list of (question, wh_word) tuples.
    """
    wh_word  = detect_answer_type(answer)
    question = apply_template(sentence, answer, wh_word)
    
    candidates = [(question, wh_word)]
    
    # Also try a "What" version if primary isn't What
    if wh_word != 'What':
        question_what = apply_template(sentence, answer, 'What')
        if question_what != question:
            candidates.append((question_what, 'What'))
    
    # Also try sentence inversion as an additional candidate
    inverted = _make_direct_question(sentence, wh_word)
    if inverted and inverted not in [q for q, _ in candidates]:
        candidates.append((inverted, wh_word))
    
    return candidates


# ─────────────────────────────────────────────────────
# STEP 3: Train Question Ranker
# ─────────────────────────────────────────────────────

def extract_question_features(question, passage):
    """
    Handcrafted features to score question quality.
    The ranker uses these to pick the best question.
    """
    words        = question.split()
    passage_words = set(passage.lower().split())
    question_words = set(question.lower().split())
    
    features = [
        # Length features
        len(words),                                          # question length
        1.0 if len(words) >= 5 else 0.0,                   # is it long enough?
        1.0 if len(words) <= 20 else 0.0,                  # is it not too long?
        
        # Wh-word features
        1.0 if question.startswith('What') else 0.0,
        1.0 if question.startswith('Who') else 0.0,
        1.0 if question.startswith('Where') else 0.0,
        1.0 if question.startswith('When') else 0.0,
        1.0 if question.startswith('Why') else 0.0,
        1.0 if question.startswith('How many') else 0.0,
        1.0 if question.startswith('How') else 0.0,
        
        # Relevance features
        len(question_words & passage_words) / (len(question_words) + 1e-9),
        
        # Grammar features
        1.0 if question.endswith('?') else 0.0,            # ends with ?
        1.0 if question[0].isupper() else 0.0,             # starts with capital
        
        # Quality features
        1.0 if '_____' in question else 0.0,               # has a blank (good)
        1.0 if 'according to' in question.lower() else 0.0, # weak fallback (bad)
        len(set(words)) / (len(words) + 1e-9),             # word diversity
    ]
    
    return features


def build_ranker_training_data(df, sample_size=5000):
    """
    Build training data for the question ranker.
    
    Positive samples (label=1): questions generated from
    the correct answer sentence (good questions)
    
    Negative samples (label=0): questions generated from
    random sentences (bad questions)
    """
    print("Building ranker training data...")
    
    X_features = []
    y_labels   = []
    
    # Sample a subset for speed
    sample_df = df.drop_duplicates(subset=['question']).head(sample_size)
    
    for _, row in sample_df.iterrows():
        passage  = str(row['article'])
        answer   = str(row['option'])
        
        # Skip if answer is empty
        if not answer.strip():
            continue
        
        # Get candidate sentences
        candidates = extract_candidate_sentences(passage, answer, top_k=3)
        
        for i, (sent, score) in enumerate(candidates):
            questions = generate_candidate_questions(sent, answer)
            
            for q, wh in questions:
                feats = extract_question_features(q, passage)
                X_features.append(feats)
                
                # Label: first candidate (highest overlap) = good question
                label = 1 if (i == 0 and score > 0.3) else 0
                y_labels.append(label)
    
    return np.array(X_features), np.array(y_labels)


def train_question_ranker(df):
    """Train SVM ranker on question quality features."""
    
    X, y = build_ranker_training_data(df, sample_size=5000)
    
    print(f"Ranker training data shape: {X.shape}")
    print(f"Label distribution: 1s={sum(y)}, 0s={len(y)-sum(y)}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    ranker = LinearSVC(class_weight='balanced', C=1.0, max_iter=2000)
    ranker.fit(X_train, y_train)
    
    y_pred = ranker.predict(X_test)
    print("\n-- Question Ranker Results --")
    print(classification_report(y_test, y_pred, 
                                target_names=["Bad Question", "Good Question"]))
    
    # Save ranker
    joblib.dump(ranker, 'models/model_a/traditional/question_ranker.pkl')
    print("Ranker saved!")
    
    return ranker


# ─────────────────────────────────────────────────────
# FULL PIPELINE: Generate question for a passage
# ─────────────────────────────────────────────────────

def generate_question(passage, correct_answer, ranker):
    """
    Full pipeline: given a passage and correct answer,
    generate the best question.
    """
    # Step 1: Get candidate sentences
    candidates = extract_candidate_sentences(passage, correct_answer, top_k=3)
    
    if not candidates:
        return f"What is the main topic of the passage?"
    
    # Step 2: Generate questions from each candidate
    all_questions = []
    for sent, overlap_score in candidates:
        questions = generate_candidate_questions(sent, correct_answer)
        for q, wh in questions:
            all_questions.append((q, sent, overlap_score))
    
    # Step 3: Rank using trained ranker
    if ranker and len(all_questions) > 1:
        feature_matrix = np.array([
            extract_question_features(q, passage) 
            for q, _, _ in all_questions
        ])
        
        scores = ranker.decision_function(feature_matrix)
        best_idx = np.argmax(scores)
        best_question = all_questions[best_idx][0]
    else:
        # Fallback: pick the one with highest overlap score
        best_question = all_questions[0][0]
    
    return best_question


# ─────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────

if __name__ == "__main__":
    
    # Load your transformed data
    df = pd.read_csv('data/transformed/train.csv')
    df['article']  = df['article'].fillna('').astype(str)
    df['question'] = df['question'].fillna('').astype(str)
    df['option']   = df['option'].fillna('').astype(str)
    
    # Train the ranker
    ranker = train_question_ranker(df)
    
    # ── Test on a sample ──────────────────────────────
    print("\n-- Testing Question Generation --")
    sample = df[df['is_correct'] == 1].drop_duplicates(subset='article').head(10)
    
    for _, row in sample.iterrows():
        passage = row['article']
        answer  = row['option']
        
        generated_q = generate_question(passage, answer, ranker)
        original_q  = row['question']
        
        print(f"\nCorrect Answer : {answer}")
        print(f"Original Q     : {original_q}")
        print(f"Generated Q    : {generated_q}")
        print("-" * 60)