import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    silhouette_score
)
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack
import warnings
warnings.filterwarnings('ignore')

# -----------------------------------------
# 1. Define Local Paths
# -----------------------------------------
# Assuming you run this from the root of 'race_rc_project'
PROJECT_DIR = os.path.abspath('.')
TransformedDataPath = os.path.join(PROJECT_DIR, 'data', 'transformed', 'train.csv')
ModelADir           = os.path.join(PROJECT_DIR, 'models', 'model_a', 'traditional')

# Add project dir to path so it can find model_a_train_generation.py locally
sys.path.append(PROJECT_DIR)

# -----------------------------------------
# 2. Re-create Feature Engineering Logic
# -----------------------------------------
def compute_handcrafted_features(df):
    features = []
    for _, row in df.iterrows():
        article_words  = set(str(row['article']).lower().split())
        question_words = set(str(row['question']).lower().split())
        option_words   = set(str(row['option']).lower().split())

        option_in_article   = len(option_words & article_words)   / (len(option_words)   + 1e-9)
        question_in_article = len(question_words & article_words)  / (len(question_words) + 1e-9)
        option_in_question  = len(option_words & question_words)   / (len(question_words) + 1e-9)
        option_length       = len(str(row['option']).split()) / 20.0
        verbatim_match      = 1.0 if str(row['option']).lower() in str(row['article']).lower() else 0.0

        features.append([
            option_in_article, question_in_article, option_in_question, option_length, verbatim_match
        ])
    return np.array(features)

def build_bow_features(df, vectorizer=None):
    option_text = df['option'].fillna('').astype(str)
    return vectorizer.transform(option_text)

# -----------------------------------------
# 3. Load Data & Prepare Test Set
# -----------------------------------------
print("Loading dataset for evaluation...")
if not os.path.exists(TransformedDataPath):
    print(f"[ERROR] Could not find {TransformedDataPath}.")
    print("Make sure you are running this script from the root of your project folder!")
    exit()

df = pd.read_csv(TransformedDataPath)

# Recreate the EXACT train/test split used during training (random_state=42)
_, X_raw_test, _, y_test = train_test_split(
    df, df['is_correct'], test_size=0.2, random_state=42, stratify=df['is_correct']
)
X_raw_test = X_raw_test.reset_index(drop=True)
y_test     = y_test.reset_index(drop=True)

# Fill NaN in text columns to prevent vectorizer errors
for col in ['article', 'question', 'option']:
    X_raw_test[col] = X_raw_test[col].fillna('')

# -----------------------------------------
# 4. Load Saved Models
# -----------------------------------------
print("Loading trained Model A artifacts...")
try:
    lr_model    = joblib.load(os.path.join(ModelADir, 'logistic_regression.pkl'))
    svm_model   = joblib.load(os.path.join(ModelADir, 'svm.pkl'))
    vectorizer  = joblib.load(os.path.join(ModelADir, 'vectorizer.pkl'))
    
    # Load the Bag-of-Words K-Means models
    kmeans_model = joblib.load(os.path.join(ModelADir, 'kmeans.pkl'))
    cluster_map  = joblib.load(os.path.join(ModelADir, 'cluster_map.pkl'))
    
except FileNotFoundError as e:
    print(f"\n[ERROR] Missing model file: {e}")
    print("Run your training script first to generate all required models.")
    exit()

# -----------------------------------------
# 5. Extract BoW+Handcrafted Features (LR & SVM)
# -----------------------------------------
print("Extracting BoW + handcrafted features for models...")
X_bow_test  = build_bow_features(X_raw_test, vectorizer=vectorizer)
X_hand_test = compute_handcrafted_features(X_raw_test)
X_test      = hstack([X_bow_test, X_hand_test]).tocsr()

# -----------------------------------------
# 6. Evaluation Helper
# -----------------------------------------
def print_rubric_metrics(model_name, y_true, y_pred):
    acc         = accuracy_score(y_true, y_pred)
    macro_f1    = f1_score(y_true, y_pred, average='macro')
    exact_match = acc   # binary classification: EM == accuracy
    cm          = confusion_matrix(y_true, y_pred)

    print(f"\n-- {model_name} --")
    print(f"Accuracy    : {acc:.4f}")
    print(f"Macro F1    : {macro_f1:.4f}")
    print(f"Exact Match : {exact_match:.4f}")
    print("Confusion Matrix:")
    print(f"TN: {cm[0][0]:<5} | FP: {cm[0][1]:<5}")
    print(f"FN: {cm[1][0]:<5} | TP: {cm[1][1]:<5}")

# -----------------------------------------
# 7. Run Evaluations
# -----------------------------------------
print("\n" + "=" * 60)
print("  MODEL A: FULL EVALUATION REPORT")
print("=" * 60)

# -- Logistic Regression --
y_pred_lr = lr_model.predict(X_test)
print_rubric_metrics("Logistic Regression", y_test, y_pred_lr)

# -- SVM --
y_pred_svm = svm_model.predict(X_test)
print_rubric_metrics("Support Vector Machine", y_test, y_pred_svm)

print(f"X_bow_test shape      : {X_bow_test.shape}")
print(f"X_hand_test shape     : {X_hand_test.shape}")
print(f"X_test shape          : {X_test.shape}")
print(f"KMeans n_features     : {kmeans_model.cluster_centers_.shape[1]}")



# -- K-Means (BoW + Handcrafted Combined Features) --
# K-Means was trained on X_train (5005 features = BoW + Handcrafted)
# so we must predict on X_test (5005) not X_bow_test (5000)
test_cluster_labels = kmeans_model.predict(X_test)
y_pred_kmeans_raw   = (test_cluster_labels == cluster_map['correct_cluster']).astype(int)
print_rubric_metrics("K-Means (BoW + Handcrafted)", y_test, y_pred_kmeans_raw)

# Silhouette score — use X_test (combined, matches training features)
rng         = np.random.default_rng(42)
sample_size = min(5000, X_test.shape[0])
sample_idx  = rng.choice(X_test.shape[0], size=sample_size, replace=False)
unique_labels = np.unique(test_cluster_labels[sample_idx])

if len(unique_labels) > 1:
    sil = silhouette_score(
        X_test[sample_idx].toarray(),      # ← X_test not X_bow_test
        test_cluster_labels[sample_idx],
        metric='cosine'
    )
    print(f"\nK-Means Silhouette Score (cosine): {sil:.4f}")
else:
    print(f"\nK-Means Silhouette Score: N/A (all points in cluster {unique_labels[0]})")

    
# -- Ensemble: LR + SVM + K-Means (Majority Vote) --
print("\n[Ensemble] Hard Voting: LR + SVM + K-Means")
y_pred_ensemble = ((y_pred_lr + y_pred_svm + y_pred_kmeans_raw) >= 2).astype(int)
print_rubric_metrics("Ensemble (Majority Vote: LR, SVM, KMeans)", y_test, y_pred_ensemble)

print("\n" + "=" * 60)

# =========================================================
#  QUESTION GENERATION EVALUATION (BLEU / ROUGE / METEOR)
# =========================================================
print("\n" + "=" * 60)
print("  QUESTION GENERATION: NLG METRICS")
print("=" * 60)

# -- Import NLG metric libraries --------------------------
import nltk
# Download required NLTK data quietly
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab', quiet=True) # Required in newer NLTK versions

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer

try:
    from model_a_train_generation import generate_question
except ImportError:
    print("[WARNING] Could not import 'generate_question'. Please ensure 'model_a_train_generation.py' is in the same folder.")
    exit()

# -- Load the trained question ranker ---------------------
print("Loading question ranker...")
ranker_path = os.path.join(ModelADir, 'question_ranker.pkl')
try:
    question_ranker = joblib.load(ranker_path)
    print(f"  Loaded ranker from {ranker_path}")
except FileNotFoundError:
    print(f"  [ERROR] Ranker not found at {ranker_path}.")
    print("  Run 'model_a_train_generation.py' first.")
    exit()

# -- Prepare evaluation sample ----------------------------
eval_df = X_raw_test[X_raw_test['is_correct'] == 1].copy() if 'is_correct' in X_raw_test.columns else df[df['is_correct'] == 1].iloc[:200].copy()

if 'is_correct' not in X_raw_test.columns:
    eval_df = X_raw_test.merge(
        df[['article', 'question', 'option', 'is_correct']],
        on=['article', 'question', 'option'], how='left'
    )
    eval_df = eval_df[eval_df['is_correct'] == 1]

eval_df = eval_df.drop_duplicates(subset='article').head(200)
print(f"  Evaluating on {len(eval_df)} unique passages...\n")

# -- Initialize scorers -----------------------------------
rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
smoother = SmoothingFunction().method1

# -- Compute scores for each sample -----------------------
bleu_scores   = []
rouge1_scores = []
rouge2_scores = []
rougeL_scores = []
meteor_scores = []

for idx, (_, row) in enumerate(eval_df.iterrows()):
    passage    = str(row['article'])
    answer     = str(row['option'])
    ref_question = str(row['question']).strip()

    # Generate question
    gen_question = generate_question(passage, answer, question_ranker)

    # Tokenize for BLEU and METEOR
    ref_tokens = word_tokenize(ref_question.lower())
    gen_tokens = word_tokenize(gen_question.lower())

    # Skip empty references
    if not ref_tokens or not gen_tokens:
        continue

    # BLEU (sentence-level with smoothing)
    bleu = sentence_bleu([ref_tokens], gen_tokens, smoothing_function=smoother)
    bleu_scores.append(bleu)

    # ROUGE
    rouge_result = rouge.score(ref_question, gen_question)
    rouge1_scores.append(rouge_result['rouge1'].fmeasure)
    rouge2_scores.append(rouge_result['rouge2'].fmeasure)
    rougeL_scores.append(rouge_result['rougeL'].fmeasure)

    # METEOR
    meteor = meteor_score([ref_tokens], gen_tokens)
    meteor_scores.append(meteor)

    # Print a few examples
    if idx < 5:
        print(f"  Example {idx+1}:")
        print(f"    Reference : {ref_question}")
        print(f"    Generated : {gen_question}")
        print(f"    BLEU={bleu:.4f}  ROUGE-L={rouge_result['rougeL'].fmeasure:.4f}  METEOR={meteor:.4f}")
        print()

# -- Aggregate and report ---------------------------------
n = len(bleu_scores)
print("-" * 60)
print(f"  Question Generation NLG Metrics  (n={n} samples)")
print("-" * 60)
print(f"  BLEU    (avg) : {np.mean(bleu_scores):.4f}")
print(f"  ROUGE-1 (avg) : {np.mean(rouge1_scores):.4f}")
print(f"  ROUGE-2 (avg) : {np.mean(rouge2_scores):.4f}")
print(f"  ROUGE-L (avg) : {np.mean(rougeL_scores):.4f}")
print(f"  METEOR  (avg) : {np.mean(meteor_scores):.4f}")
print("-" * 60)

print("\n" + "=" * 60)
print("  EVALUATION COMPLETE")
print("=" * 60)