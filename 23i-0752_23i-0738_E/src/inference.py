import os
import numpy as np
import pandas as pd
import joblib
from scipy.sparse import hstack
import warnings
warnings.filterwarnings('ignore')

# -----------------------------------------
# Load saved models (done once at startup)
# -----------------------------------------
ModelsDir = 'models/model_a/traditional'

print("Loading trained models into memory...")
lr_model    = joblib.load(os.path.join(ModelsDir, 'logistic_regression.pkl'))
svm_model   = joblib.load(os.path.join(ModelsDir, 'svm.pkl'))
vectorizer  = joblib.load(os.path.join(ModelsDir, 'vectorizer.pkl'))

# K-Means trained on BoW + Handcrafted (5005 features)
kmeans_model        = joblib.load(os.path.join(ModelsDir, 'kmeans.pkl'))
cluster_map         = joblib.load(os.path.join(ModelsDir, 'cluster_map.pkl'))
correct_cluster_idx = cluster_map['correct_cluster']

print("All models loaded successfully!")

# -----------------------------------------
# Handcrafted features
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
            option_in_article,
            question_in_article,
            option_in_question,
            option_length,
            verbatim_match
        ])
    return np.array(features)


# -----------------------------------------
# Core feature builder (shared by all models)
# -----------------------------------------
def build_features(rows):
    """
    Build the combined 5005-feature matrix for a DataFrame of rows.
    Same pipeline as training: BoW (5000) + Handcrafted (5).
    """
    X_bow  = vectorizer.transform(rows['option'].fillna('').astype(str))
    X_hand = compute_handcrafted_features(rows)
    return hstack([X_bow, X_hand]).tocsr()


# -----------------------------------------
# Core prediction function
# -----------------------------------------
def predict_answer(article, question, choices):
    """
    Given an article, a question, and a list of 4 choices (A-D),
    returns the predicted correct choice for all 3 models.
    """
    labels = ['A', 'B', 'C', 'D']

    # Build one row per choice
    rows = pd.DataFrame([{
        'article':  article,
        'question': question,
        'option':   str(choice)
    } for choice in choices])

    # Fill NaN safety
    for col in ['article', 'question', 'option']:
        rows[col] = rows[col].fillna('').astype(str)

    # Build combined features (5005) — used by ALL three models
    X = build_features(rows)

    results = {}

    # ─────────────────────────────────────────
    # 1. Logistic Regression
    # ─────────────────────────────────────────
    lr_scores  = lr_model.predict_proba(X)[:, 1]   # P(Correct)
    best_lr    = int(np.argmax(lr_scores))

    results["Logistic Regression"] = {
        'predicted_label': labels[best_lr],
        'predicted_text':  choices[best_lr],
        'score_type':      "Confidence Probability (higher = better)",
        'scores':          {labels[i]: round(float(lr_scores[i]), 4)
                            for i in range(len(choices))}
    }

    # ─────────────────────────────────────────
    # 2. SVM
    # ─────────────────────────────────────────
    svm_scores = svm_model.decision_function(X)     # distance from hyperplane
    best_svm   = int(np.argmax(svm_scores))

    results["SVM"] = {
        'predicted_label': labels[best_svm],
        'predicted_text':  choices[best_svm],
        'score_type':      "Decision Function Score (higher = better)",
        'scores':          {labels[i]: round(float(svm_scores[i]), 4)
                            for i in range(len(choices))}
    }

    # ─────────────────────────────────────────
    # 3. K-Means (BoW + Handcrafted)
    # Uses distance to the "Correct" centroid
    # Lowest distance = most likely correct answer
    # ─────────────────────────────────────────
    all_distances   = kmeans_model.transform(X)          # shape (4, 2)
    dist_to_correct = all_distances[:, correct_cluster_idx]
    best_kmeans     = int(np.argmin(dist_to_correct))    # closest to correct centroid

    results["K-Means (BoW + Handcrafted)"] = {
        'predicted_label': labels[best_kmeans],
        'predicted_text':  choices[best_kmeans],
        'score_type':      "Distance to 'Correct' Centroid (lower = better)",
        'scores':          {labels[i]: round(float(dist_to_correct[i]), 4)
                            for i in range(len(choices))}
    }

    # ─────────────────────────────────────────
    # 4. Ensemble: Majority Vote (LR + SVM + KMeans)
    # ─────────────────────────────────────────
    votes = [best_lr, best_svm, best_kmeans]

    # Count votes per option index
    vote_counts = np.bincount(votes, minlength=len(choices))
    best_ensemble = int(np.argmax(vote_counts))

    results["Ensemble (Majority Vote)"] = {
        'predicted_label': labels[best_ensemble],
        'predicted_text':  choices[best_ensemble],
        'score_type':      "Vote Count (higher = more models agree)",
        'scores':          {labels[i]: int(vote_counts[i])
                            for i in range(len(choices))}
    }

    return results


# -----------------------------------------
# Pretty printer
# -----------------------------------------
def print_prediction(article, question, choices, results):
    labels = ['A', 'B', 'C', 'D']
    sep    = "=" * 65

    print(f"\n{sep}")
    print("  ARTICLE (first 200 chars):")
    print(f"  {article[:200].strip()}...")
    print(f"\n  QUESTION: {question}")
    print("\n  CHOICES:")
    for i, c in enumerate(choices):
        print(f"    {labels[i]}) {c}")
    print(sep)

    for model_name, res in results.items():
        print(f"\n  [{model_name}]")
        print(f"  Predicted answer : {res['predicted_label']}) {res['predicted_text']}")
        print(f"  Metric Type      : {res['score_type']}")
        print(f"  Itemized Scores  :")
        for label, score in res['scores'].items():
            print(f"    {label}: {score}")

    print(f"\n{sep}\n")


# -----------------------------------------
# Example usage
# -----------------------------------------
if __name__ == '__main__':

    # --- TEST 1 ---
    article = """
    The Amazon rainforest, also known as Amazonia, is a moist broadleaf tropical rainforest
    in the Amazon biome that covers most of the Amazon basin of South America. This basin
    encompasses 7,000,000 km2, of which 5,500,000 km2 are covered by the rainforest.
    This region includes territory belonging to nine nations and 3,344 formally acknowledged
    indigenous territories. The majority of the forest is contained within Brazil,
    with 60% of the rainforest, followed by Peru with 13%, Colombia with 10%,
    and with minor amounts in Bolivia, Ecuador, French Guiana, Guyana, Suriname, and Venezuela.
    """

    question = "What percentage of the Amazon rainforest is contained within Brazil?"
    choices  = ["13%", "60%", "10%", "30%"]

    results = predict_answer(article, question, choices)
    print_prediction(article, question, choices, results)

    # --- TEST 2 ---
    article2 = """
    In 2025, EcoMotors introduced the 'AeroDrive' hybrid vehicle, which combines a traditional
    combustion engine with a new aerodynamic solar-panel roof. Unlike previous models that
    only charged while parked, the AeroDrive can convert solar energy into battery power
    while driving at speeds over 40 mph. This innovation extends the car's range by up to 15%
    on sunny days, making it highly popular among long-distance commuters in sunny states
    like California and Arizona.
    """

    question2 = "What is the primary advantage of the AeroDrive's solar-panel roof?"
    choices2  = [
        "It allows the car to run entirely on solar energy without using gas.",
        "It makes the car 15% faster when driving on sunny days.",
        "It can charge the battery while the car is moving at certain speeds.",
        "It was specifically invented for commuters in California and Arizona."
    ]

    results2 = predict_answer(article2, question2, choices2)
    print_prediction(article2, question2, choices2, results2)