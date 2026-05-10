import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack
import joblib
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

TransformedDataPath = 'data/transformed/train.csv'
FeaturePath = 'data/transformed/features.npz'
MODEL_DIR = 'models/model_a/traditional'
# -----------------------------------------
# STEP 1: Handcrafted Relational Features
# -----------------------------------------
def compute_handcrafted_features(df):
    """
    These features capture the RELATIONSHIP between
    article, question, and option -- not just raw text.
    """
    features = []

    for _, row in df.iterrows():
        article_words  = set(str(row['article']).lower().split())
        question_words = set(str(row['question']).lower().split())
        option_words   = set(str(row['option']).lower().split())

        # How many option words appear in the article?
        option_in_article = len(option_words & article_words) / (len(option_words) + 1e-9)

        # How many question words appear in the article?
        question_in_article = len(question_words & article_words) / (len(question_words) + 1e-9)

        # How many option words appear in the question?
        option_in_question = len(option_words & question_words) / (len(question_words) + 1e-9)

        # Length of the option (normalized)
        option_length = len(str(row['option']).split()) / 20.0

        # Does the option appear verbatim in the article?
        verbatim_match = 1.0 if str(row['option']).lower() in str(row['article']).lower() else 0.0

        features.append([
            option_in_article,
            question_in_article,
            option_in_question,
            option_length,
            verbatim_match
        ])

    return np.array(features)


# -----------------------------------------
# STEP 2: Bag-of-Words on Option Text Only
# (This is your One-Hot / BoW representation)
# -----------------------------------------
def build_bow_features(df, vectorizer=None, fit=True):
    """
    One-Hot / Bag-of-Words on just the option text.
    We keep article separate to avoid drowning the signal.
    """
    option_text = df['option'].astype(str)

    if fit:
        vectorizer = CountVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=3,
            binary=True  # This makes it true One-Hot Encoding
        )
        X_bow = vectorizer.fit_transform(option_text)
    else:
        X_bow = vectorizer.transform(option_text)

    return X_bow, vectorizer


def transformData():
    if os.path.exists(TransformedDataPath):
        print("Loading existing transformed data...")
        df = pd.read_csv(TransformedDataPath)
        return df

    print("Reading raw dataset...")
    train_df = pd.read_csv('data/processed/train.csv')

    # The new dataset has separate A, B, C, D columns directly
    option_cols = ['A', 'B', 'C', 'D']
    answer_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    reshaped_records = []

    print(f"Transforming data ({len(train_df)} rows)...")
    for row in train_df.itertuples(index=False):
        correct_idx = answer_map.get(str(row.answer).strip())
        if correct_idx is None:
            continue

        for i, col in enumerate(option_cols):
            option_text = str(getattr(row, col, ''))
            reshaped_records.append({
                'article':    str(row.article),
                'question':   str(row.question),
                'option':     option_text,
                'is_correct': 1 if i == correct_idx else 0
            })

    df = pd.DataFrame(reshaped_records)
    os.makedirs('data/transformed', exist_ok=True)
    df.to_csv(TransformedDataPath, index=False)
    print(f"Saved! Shape: {df.shape}")
    return df


# -----------------------------------------
# MAIN
# -----------------------------------------
df = transformData()

print(f"\nClass distribution:\n{df['is_correct'].value_counts()}")

# Split BEFORE feature extraction to avoid data leakage
X_raw_train, X_raw_test, y_train, y_test = train_test_split(
    df, df['is_correct'],
    test_size=0.2,
    random_state=42,
    stratify=df['is_correct']   # keep 25/75 ratio in both splits
)
X_raw_train = X_raw_train.reset_index(drop=True)
X_raw_test  = X_raw_test.reset_index(drop=True)

# Force all text columns to string, no NaN can survive this
X_raw_train['option']   = X_raw_train['option'].fillna('').astype(str)
X_raw_train['article']  = X_raw_train['article'].fillna('').astype(str)
X_raw_train['question'] = X_raw_train['question'].fillna('').astype(str)

X_raw_test['option']   = X_raw_test['option'].fillna('').astype(str)
X_raw_test['article']  = X_raw_test['article'].fillna('').astype(str)
X_raw_test['question'] = X_raw_test['question'].fillna('').astype(str)

print("\nBuilding features...")

# 1. Bag-of-Words (One-Hot style) features
X_bow_train, vectorizer = build_bow_features(X_raw_train, fit=True)
X_bow_test, _           = build_bow_features(X_raw_test,  vectorizer=vectorizer, fit=False)

# 2. Handcrafted relational features
print("Computing handcrafted features (this takes a minute)...")
X_hand_train = compute_handcrafted_features(X_raw_train)
X_hand_test  = compute_handcrafted_features(X_raw_test)

# 3. Combine both feature sets
X_train = hstack([X_bow_train, X_hand_train]).tocsr()
X_test  = hstack([X_bow_test,  X_hand_test]).tocsr()

print(f"Final feature matrix shape: {X_train.shape}")

# -----------------------------------------
# Train Logistic Regression
# -----------------------------------------
print("\nTraining Logistic Regression...")
lr_model = LogisticRegression(
    class_weight='balanced',  # handles the 75/25 imbalance
    C=1.0,
    max_iter=1000,
    solver='saga',            # faster for large sparse data
    n_jobs=-1
)
lr_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)
print("\n-- Logistic Regression Results --")
print(classification_report(y_test, y_pred_lr, target_names=["Incorrect", "Correct"]))

# -----------------------------------------
# Train SVM
# -----------------------------------------
print("Training SVM...")
svm_model = LinearSVC(
    class_weight='balanced',
    C=1.0,
    max_iter=2000
)
svm_model.fit(X_train, y_train)

y_pred_svm = svm_model.predict(X_test)
print("\n-- SVM Results --")
print(classification_report(y_test, y_pred_svm, target_names=["Incorrect", "Correct"]))

# -----------------------------------------
# Save models
# -----------------------------------------
os.makedirs('models/model_a/traditional', exist_ok=True)
joblib.dump(lr_model,   'models/model_a/traditional/logistic_regression.pkl')
joblib.dump(svm_model,  'models/model_a/traditional/svm.pkl')
joblib.dump(vectorizer, 'models/model_a/traditional/vectorizer.pkl')
print("\nModels saved!")

print("\n" + "=" * 55)
print("  STEP 4: Unsupervised Clustering (K-Means)")
print("  Using BoW + Handcrafted Combined Features")
print("=" * 55)

# ─────────────────────────────────────────────────────
# KEY FIX: Use combined features (X_train = BoW + Handcrafted)
# Option text alone doesn't separate correct/incorrect answers.
# The RELATIONAL features (overlap with article, verbatim match)
# are what actually distinguish correct from incorrect options.
# ─────────────────────────────────────────────────────

print("Running K-Means (K=2) on combined features...")

kmeans = MiniBatchKMeans(
    n_clusters=2,
    random_state=42,
    batch_size=1024,
    max_iter=300,
    n_init=10
)

# Fit on COMBINED features (BoW + Handcrafted = 5005)
# This is the same X_train used by LR and SVM
kmeans.fit(X_train)
train_labels = kmeans.labels_

print(f"\nK-Means trained on: {kmeans.cluster_centers_.shape[1]} features")

# ─────────────────────────────────────────────────────
# Analyse cluster composition
# ─────────────────────────────────────────────────────
print("\nAnalysing cluster composition...")

for c in [0, 1]:
    mask        = (train_labels == c)
    total       = mask.sum()
    n_correct   = (y_train[mask] == 1).sum()
    n_incorrect = (y_train[mask] == 0).sum()
    purity      = n_correct / total * 100 if total > 0 else 0
    print(f"  Cluster {c}: {total:>6} rows  |  "
          f"Correct: {n_correct}  Incorrect: {n_incorrect}  |  "
          f"Purity: {purity:.1f}%")

purity_0 = (y_train[train_labels == 0] == 1).mean()
purity_1 = (y_train[train_labels == 1] == 1).mean()
correct_cluster   = 0 if purity_0 > purity_1 else 1
incorrect_cluster = 1 - correct_cluster
print(f"\n  --> Cluster {correct_cluster} identified as the 'CORRECT' cluster")

def cluster_purity(labels, true_labels):
    total = 0
    for c in np.unique(labels):
        mask     = (labels == c)
        majority = np.bincount(true_labels[mask]).max()
        total   += majority
    return total / len(labels)

train_purity = cluster_purity(train_labels, y_train.values)
print(f"\n  Overall cluster purity (train) : {train_purity:.4f}")

# ─────────────────────────────────────────────────────
# Silhouette Score
# Now uses combined features — should give meaningful score
# ─────────────────────────────────────────────────────
print("\n  Computing silhouette score on subset...")

sil_size  = min(5000, X_train.shape[0])
rng_sil   = np.random.default_rng(42)
sil_idx   = rng_sil.choice(X_train.shape[0], size=sil_size, replace=False)

unique_sil_labels = np.unique(train_labels[sil_idx])

if len(unique_sil_labels) > 1:
    sil_score = silhouette_score(
        X_train[sil_idx].toarray(),   # combined features
        train_labels[sil_idx],
        metric='cosine'
    )
    print(f"  Silhouette score (cosine) : {sil_score:.4f}")
else:
    sil_score = -1.0
    print("  Silhouette score          : N/A (only 1 cluster in sample)")

# ─────────────────────────────────────────────────────
# MCQ Accuracy on Test Set
# ─────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  MCQ Accuracy Evaluation")
print("=" * 55)

# Use combined X_test (5005 features — matches what KMeans was trained on)
distances       = kmeans.transform(X_test)
dist_to_correct = distances[:, correct_cluster]

X_raw_test['_true_label']      = y_test.values
X_raw_test['_dist_to_correct'] = dist_to_correct

groups = X_raw_test.groupby(['article', 'question'], sort=False)

total_questions = 0
correct_answers = 0
skipped_groups  = 0

for _, group in groups:
    true_labels_grp = group['_true_label'].values
    if 1 not in true_labels_grp:
        skipped_groups += 1
        continue
    best_idx = group['_dist_to_correct'].values.argmin()
    if true_labels_grp[best_idx] == 1:
        correct_answers += 1
    total_questions += 1

mcq_accuracy = correct_answers / total_questions * 100 if total_questions > 0 else 0
print(f"\n  Questions evaluated : {total_questions}")
print(f"  Correctly answered  : {correct_answers}")
print(f"  MCQ Accuracy        : {mcq_accuracy:.2f}%")
if skipped_groups > 0:
    print(f"  [!] Skipped groups  : {skipped_groups}")

# Binary classification report
test_labels = kmeans.predict(X_test)
binary_pred = (test_labels == correct_cluster).astype(int)

print("\n  Binary Classification Report:")
print(classification_report(y_test, binary_pred,
                             target_names=["Incorrect", "Correct"]))

# ─────────────────────────────────────────────────────
# Visualization: TruncatedSVD
# ─────────────────────────────────────────────────────
print("Generating cluster visualisation...")

rng2     = np.random.default_rng(0)
vis_size = min(3000, X_train.shape[0])
vis_idx  = rng2.choice(X_train.shape[0], size=vis_size, replace=False)

svd   = TruncatedSVD(n_components=2, random_state=42)
X_2d  = svd.fit_transform(X_train[vis_idx])   # combined features
labels_2d = train_labels[vis_idx]
true_2d   = y_train.values[vis_idx]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    'K-Means Clustering of Question-Answer Pairs\n'
    '(BoW + Handcrafted Features, TruncatedSVD to 2D)',
    fontsize=13
)

c_map = {0: '#2196F3', 1: '#FF5722'}
axes[0].scatter(X_2d[:, 0], X_2d[:, 1],
                c=[c_map[l] for l in labels_2d], alpha=0.4, s=8)
axes[0].set_title('Coloured by K-Means Cluster')
axes[0].set_xlabel('Component 1')
axes[0].set_ylabel('Component 2')
axes[0].legend(handles=[
    Patch(color='#2196F3', label='Cluster 0'),
    Patch(color='#FF5722', label='Cluster 1')
])

t_map = {1: '#4CAF50', 0: '#9E9E9E'}
axes[1].scatter(X_2d[:, 0], X_2d[:, 1],
                c=[t_map[l] for l in true_2d], alpha=0.4, s=8)
axes[1].set_title('Coloured by True Label')
axes[1].set_xlabel('Component 1')
axes[1].set_ylabel('Component 2')
axes[1].legend(handles=[
    Patch(color='#4CAF50', label='Correct'),
    Patch(color='#9E9E9E', label='Incorrect')
])

plt.tight_layout()
plot_path = os.path.join(MODEL_DIR, 'cluster_visualisation_bow.png')
plt.savefig(plot_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"  Saved: {plot_path}")

# ─────────────────────────────────────────────────────
# Save models
# ─────────────────────────────────────────────────────
joblib.dump(kmeans,
            os.path.join(MODEL_DIR, 'kmeans.pkl'))
joblib.dump({'correct_cluster':   correct_cluster,
             'incorrect_cluster': incorrect_cluster},
             os.path.join(MODEL_DIR, 'cluster_map.pkl'))

print("\n" + "=" * 55)
print("  FINAL SUMMARY")
print("=" * 55)
print(f"  Algorithm          : K-Means (K=2, MiniBatch)")
print(f"  Features           : BoW + Handcrafted (5005)")
print(f"  Cluster purity     : {train_purity:.4f}")
print(f"  Silhouette (cosine): {sil_score:.4f}")
print(f"  MCQ Accuracy       : {mcq_accuracy:.2f}%")
print("=" * 55)
