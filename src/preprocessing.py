import os
import joblib
import scipy.sparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import hstack, csr_matrix
import string

# ─────────────────────────────────────────
# Paths
# ─────────────────────────────────────────
RAW_TRAIN_PATH  = 'archive/train.csv'
RAW_DEV_PATH    = 'archive/dev.csv'
RAW_TEST_PATH   = 'archive/test.csv'

PROCESSED_DIR       = 'data/processed'
PROCESSED_TRAIN     = os.path.join(PROCESSED_DIR, 'train.csv')
PROCESSED_DEV       = os.path.join(PROCESSED_DIR, 'dev.csv')
PROCESSED_TEST      = os.path.join(PROCESSED_DIR, 'test.csv')

EDA_DIR             = 'data/eda'

VECTORIZER_PATH     = 'models/tfidf_vectorizer.pkl'
SCALER_PATH         = 'models/scaler.pkl'
MATRIX_PATH         = 'models/X_train.npz'


def remove_punctuation(text):
    """Remove all punctuation from text."""
    table = str.maketrans('', '', string.punctuation)
    return text.translate(table)


# ═════════════════════════════════════════
#  STEP 1: Load & Clean the RACE Dataset
# ═════════════════════════════════════════
def load_and_clean(csv_path):
    """
    Load a RACE CSV split from the archive/ folder.
    Columns expected: Unnamed: 0, id, article, question, A, B, C, D, answer
    Returns a cleaned DataFrame with an 'options' list column and 'difficulty' column.
    """
    print(f"  Loading {csv_path}...")
    df = pd.read_csv(csv_path)

    # Drop the unnecessary index column
    if 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'], inplace=True)

    # Fill any null option values with empty string
    for col in ['A', 'B', 'C', 'D']:
        df[col] = df[col].fillna('').astype(str)

    df['article']  = df['article'].fillna('').astype(str)
    df['question'] = df['question'].fillna('').astype(str)
    df['answer']   = df['answer'].fillna('').astype(str).str.strip()

    # Create a combined 'options' list column for downstream compatibility
    df['options'] = df.apply(lambda row: [row['A'], row['B'], row['C'], row['D']], axis=1)

    # Extract difficulty level from the ID (e.g., "middle7348.txt" → "middle", "high3456.txt" → "high")
    df['difficulty'] = df['id'].apply(
        lambda x: 'middle' if str(x).startswith('middle') else ('high' if str(x).startswith('high') else 'unknown')
    )

    # Lexical features on the article text
    df['word_count']      = df['article'].apply(lambda x: len(x.split()))
    df['char_count']      = df['article'].apply(lambda x: len(x))
    df['avg_word_length'] = df['char_count'] / (df['word_count'] + 1)

    print(f"    -> Shape: {df.shape}")
    return df


# ═════════════════════════════════════════
#  STEP 2: Exploratory Data Analysis (EDA)
# ═════════════════════════════════════════
def run_eda(train_df, dev_df, test_df):
    """
    Generate EDA visualizations and summary statistics as required:
      - Passage length distributions
      - Question type analysis
      - Answer balance
      - Difficulty level breakdown
      - Summary statistics table
    All plots are saved to data/eda/
    """
    os.makedirs(EDA_DIR, exist_ok=True)
    print("\n" + "=" * 55)
    print("  EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 55)

    # ── 1. Summary Statistics Table ────────────────────────
    print("\n── Summary Statistics ──")
    summary = {
        'Split':             ['Train', 'Dev', 'Test'],
        'Total Questions':   [len(train_df), len(dev_df), len(test_df)],
        'Unique Articles':   [train_df['article'].nunique(), dev_df['article'].nunique(), test_df['article'].nunique()],
        'Avg Passage Words': [round(train_df['word_count'].mean(), 1), round(dev_df['word_count'].mean(), 1), round(test_df['word_count'].mean(), 1)],
        'Avg Passage Chars': [round(train_df['char_count'].mean(), 1), round(dev_df['char_count'].mean(), 1), round(test_df['char_count'].mean(), 1)],
        'Middle Level':      [(train_df['difficulty'] == 'middle').sum(), (dev_df['difficulty'] == 'middle').sum(), (test_df['difficulty'] == 'middle').sum()],
        'High Level':        [(train_df['difficulty'] == 'high').sum(), (dev_df['difficulty'] == 'high').sum(), (test_df['difficulty'] == 'high').sum()],
    }
    summary_df = pd.DataFrame(summary)
    print(summary_df.to_string(index=False))
    summary_df.to_csv(os.path.join(EDA_DIR, 'summary_statistics.csv'), index=False)

    # ── 2. Answer Balance Distribution ────────────────────
    print("\n── Answer Balance (Train) ──")
    answer_counts = train_df['answer'].value_counts().sort_index()
    print(answer_counts.to_string())

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#4CAF50', '#2196F3', '#FF9800', '#E91E63']
    bars = ax.bar(answer_counts.index, answer_counts.values, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_title('Answer Distribution (Train Split)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Answer Choice', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    for bar, count in zip(bars, answer_counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
                f'{count}\n({count/len(train_df)*100:.1f}%)', ha='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, 'answer_distribution.png'), dpi=150)
    plt.close()
    print(f"  Saved: {EDA_DIR}/answer_distribution.png")

    # ── 3. Passage Length Distribution ─────────────────────
    print("\n── Passage Length Distribution (Train) ──")
    print(f"  Min words  : {train_df['word_count'].min()}")
    print(f"  Max words  : {train_df['word_count'].max()}")
    print(f"  Mean words : {train_df['word_count'].mean():.1f}")
    print(f"  Median     : {train_df['word_count'].median():.0f}")
    print(f"  Std Dev    : {train_df['word_count'].std():.1f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Passage Length Analysis (Train Split)', fontsize=14, fontweight='bold')

    # Word count histogram
    axes[0].hist(train_df['word_count'], bins=50, color='#2196F3', edgecolor='black', alpha=0.7)
    axes[0].axvline(train_df['word_count'].mean(), color='red', linestyle='--', label=f"Mean: {train_df['word_count'].mean():.0f}")
    axes[0].axvline(train_df['word_count'].median(), color='green', linestyle='--', label=f"Median: {train_df['word_count'].median():.0f}")
    axes[0].set_title('Word Count Distribution')
    axes[0].set_xlabel('Word Count')
    axes[0].set_ylabel('Frequency')
    axes[0].legend()

    # Character count histogram
    axes[1].hist(train_df['char_count'], bins=50, color='#FF9800', edgecolor='black', alpha=0.7)
    axes[1].axvline(train_df['char_count'].mean(), color='red', linestyle='--', label=f"Mean: {train_df['char_count'].mean():.0f}")
    axes[1].set_title('Character Count Distribution')
    axes[1].set_xlabel('Character Count')
    axes[1].set_ylabel('Frequency')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, 'passage_length_distribution.png'), dpi=150)
    plt.close()
    print(f"  Saved: {EDA_DIR}/passage_length_distribution.png")

    # ── 4. Difficulty Level Breakdown ──────────────────────
    print("\n── Difficulty Level Breakdown (Train) ──")
    diff_counts = train_df['difficulty'].value_counts()
    print(diff_counts.to_string())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Difficulty Level Analysis (Train Split)', fontsize=14, fontweight='bold')

    diff_colors = {'middle': '#4CAF50', 'high': '#E91E63', 'unknown': '#9E9E9E'}
    ordered_labels = [l for l in ['middle', 'high', 'unknown'] if l in diff_counts.index]
    ordered_values = [diff_counts[l] for l in ordered_labels]
    ordered_colors = [diff_colors[l] for l in ordered_labels]

    # Pie chart
    axes[0].pie(ordered_values, labels=ordered_labels, colors=ordered_colors,
                autopct='%1.1f%%', startangle=90, textprops={'fontsize': 12})
    axes[0].set_title('Difficulty Distribution')

    # Passage length by difficulty (box plot)
    middle_wc = train_df[train_df['difficulty'] == 'middle']['word_count']
    high_wc   = train_df[train_df['difficulty'] == 'high']['word_count']
    bp = axes[1].boxplot([middle_wc, high_wc], labels=['Middle School', 'High School'],
                          patch_artist=True)
    bp['boxes'][0].set_facecolor('#4CAF50')
    bp['boxes'][1].set_facecolor('#E91E63')
    axes[1].set_title('Passage Length by Difficulty')
    axes[1].set_ylabel('Word Count')

    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, 'difficulty_breakdown.png'), dpi=150)
    plt.close()
    print(f"  Saved: {EDA_DIR}/difficulty_breakdown.png")

    # ── 5. Question Type Analysis ──────────────────────────
    print("\n── Question Type Analysis (Train) ──")

    def classify_question_type(q):
        q_lower = q.lower().strip()
        if q_lower.startswith('what'):   return 'What'
        if q_lower.startswith('which'):  return 'Which'
        if q_lower.startswith('who'):    return 'Who'
        if q_lower.startswith('where'):  return 'Where'
        if q_lower.startswith('when'):   return 'When'
        if q_lower.startswith('why'):    return 'Why'
        if q_lower.startswith('how'):    return 'How'
        # Check for fill-in-the-blank style (contains _ or blank)
        if '_' in q or 'blank' in q_lower:  return 'Fill-in-Blank'
        return 'Other'

    train_df['question_type'] = train_df['question'].apply(classify_question_type)
    qtype_counts = train_df['question_type'].value_counts()
    print(qtype_counts.to_string())

    fig, ax = plt.subplots(figsize=(10, 6))
    q_colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0',
                '#00BCD4', '#FF5722', '#607D8B', '#795548']
    bars = ax.barh(qtype_counts.index[::-1], qtype_counts.values[::-1],
                   color=q_colors[:len(qtype_counts)], edgecolor='black', linewidth=0.5)
    ax.set_title('Question Type Distribution (Train Split)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Count', fontsize=12)
    for bar, count in zip(bars, qtype_counts.values[::-1]):
        ax.text(bar.get_width() + 200, bar.get_y() + bar.get_height() / 2,
                f'{count} ({count/len(train_df)*100:.1f}%)', va='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, 'question_type_distribution.png'), dpi=150)
    plt.close()
    print(f"  Saved: {EDA_DIR}/question_type_distribution.png")

    # ── 6. Answer Balance by Difficulty ────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Answer Balance by Difficulty Level (Train)', fontsize=14, fontweight='bold')

    for i, level in enumerate(['middle', 'high']):
        subset = train_df[train_df['difficulty'] == level]
        ans_counts = subset['answer'].value_counts().sort_index()
        axes[i].bar(ans_counts.index, ans_counts.values, color=colors, edgecolor='black', linewidth=0.5)
        axes[i].set_title(f'{level.title()} School (n={len(subset)})')
        axes[i].set_xlabel('Answer Choice')
        axes[i].set_ylabel('Count')
        for idx_bar, (label, count) in enumerate(zip(ans_counts.index, ans_counts.values)):
            axes[i].text(idx_bar, count + 100, f'{count}', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, 'answer_by_difficulty.png'), dpi=150)
    plt.close()
    print(f"  Saved: {EDA_DIR}/answer_by_difficulty.png")

    # ── 7. Option Length Analysis ──────────────────────────
    print("\n── Option Length Statistics (Train) ──")
    for col in ['A', 'B', 'C', 'D']:
        avg_len = train_df[col].apply(lambda x: len(str(x).split())).mean()
        print(f"  Avg words in Option {col}: {avg_len:.1f}")

    print(f"\n  EDA complete! All plots saved to '{EDA_DIR}/'")
    print("=" * 55)

    return train_df


# ═════════════════════════════════════════
#  STEP 3: Save Processed Data
# ═════════════════════════════════════════
def save_processed(train_df, dev_df, test_df):
    """Save the cleaned and enriched DataFrames for downstream use."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    train_df.to_csv(PROCESSED_TRAIN, index=False)
    dev_df.to_csv(PROCESSED_DEV, index=False)
    test_df.to_csv(PROCESSED_TEST, index=False)
    print(f"\nProcessed data saved to '{PROCESSED_DIR}/'")


# ═════════════════════════════════════════
#  STEP 4: TF-IDF Vectorization + Lexical Features
# ═════════════════════════════════════════
def build_features(train_df):
    """
    Build TF-IDF + lexical feature matrix on article text.
    Saves: vectorizer, scaler, combined feature matrix.
    """
    print("\n" + "=" * 55)
    print("  FEATURE ENGINEERING")
    print("=" * 55)

    corpus = train_df['article'].tolist()

    # 1. TF-IDF Vectorization
    print("\nFitting TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=10000,       # Vocabulary size
        stop_words='english',     # Remove common English stopwords
        sublinear_tf=True,        # Use log(1+TF) to dampen high frequencies
        ngram_range=(1, 2),       # Include unigrams and bigrams
        min_df=2,                 # Ignore very rare terms
        max_df=0.95,              # Ignore near-universal terms
    )
    X_train_tfidf = vectorizer.fit_transform(corpus)
    print(f"  TF-IDF matrix shape: {X_train_tfidf.shape}")

    # 2. Lexical Features
    lexical_features = train_df[['word_count', 'char_count', 'avg_word_length']].values
    print(f"  Lexical features shape: {lexical_features.shape}")

    # 3. Scale the Lexical Features to match TF-IDF ranges (0 to 1)
    scaler = MinMaxScaler()
    lexical_scaled = scaler.fit_transform(lexical_features)

    # 4. Combine them using hstack (Horizontal Stack)
    X_train_final = hstack([X_train_tfidf, csr_matrix(lexical_scaled)])
    print(f"  Combined feature matrix shape: {X_train_final.shape}")

    # 5. Save the artifacts for future runs
    os.makedirs('models', exist_ok=True)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(scaler, SCALER_PATH)
    scipy.sparse.save_npz(MATRIX_PATH, X_train_final)
    print(f"\n  Saved: {VECTORIZER_PATH}")
    print(f"  Saved: {SCALER_PATH}")
    print(f"  Saved: {MATRIX_PATH}")

    return X_train_final, vectorizer, scaler


def demo_cosine_similarity(X_train_final, vectorizer):
    """Quick demo: find the most similar article to a target document."""
    print("\n── Cosine Similarity Demo ──")
    target_vector = X_train_final[40]
    similarity_scores = cosine_similarity(target_vector, X_train_final)
    scores_1d = similarity_scores.flatten()
    most_similar_index = np.argsort(scores_1d)[-2]  # -1 is itself, -2 is most similar
    print(f"  Target document index     : 40")
    print(f"  Most similar article index: {most_similar_index}")
    print(f"  Similarity Score          : {scores_1d[most_similar_index]:.4f}")


# ═════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════
if __name__ == '__main__':

    # Check if features are already built (skip to loading)
    if os.path.exists(VECTORIZER_PATH) and os.path.exists(MATRIX_PATH):
        print('Loading pre-computed vectorizer and matrix from disk...')
        vectorizer     = joblib.load(VECTORIZER_PATH)
        X_train_final  = scipy.sparse.load_npz(MATRIX_PATH)

        demo_cosine_similarity(X_train_final, vectorizer)

        print(f'\nFinal Matrix shape : {X_train_final.shape}')
        print(f'Vocabulary sample  : {list(vectorizer.vocabulary_.items())[:5]}')

    else:
        # ── Load raw data from archive/ ──────────────────
        print("=" * 55)
        print("  LOADING RACE DATASET")
        print("=" * 55)
        train_df = load_and_clean(RAW_TRAIN_PATH)
        dev_df   = load_and_clean(RAW_DEV_PATH)
        test_df  = load_and_clean(RAW_TEST_PATH)

        # ── Run EDA ──────────────────────────────────────
        train_df = run_eda(train_df, dev_df, test_df)

        # ── Save processed data ──────────────────────────
        save_processed(train_df, dev_df, test_df)

        # ── Build feature matrices ───────────────────────
        X_train_final, vectorizer, scaler = build_features(train_df)

        # ── Demo ─────────────────────────────────────────
        demo_cosine_similarity(X_train_final, vectorizer)

        print(f'\nFinal Matrix shape : {X_train_final.shape}')
        print(f'Vocabulary sample  : {list(vectorizer.vocabulary_.items())[:5]}')

    print("\n[DONE] Preprocessing complete!")
