# 🧠 AI-Powered Quiz Engine & Automated Question Generation

This repository contains an end-to-end Machine Learning pipeline and interactive Streamlit Dashboard designed to automatically generate, verify, and evaluate reading comprehension questions from text articles. The system leverages both classical Machine Learning models and NLP techniques to construct challenging distractors, verify answers, and generate contextual hints.

---

## 📌 Key Features

1. **Answer Verification (Model A)**: Evaluates a given question and set of options against an article to determine the correct answer. Uses Logistic Regression, SVM, K-Means clustering, and a Majority-Vote Ensemble model.
2. **Distractor Generation (Model B)**: Autonomously generates plausible but incorrect multiple-choice options (distractors) given an article, question, and correct answer.
3. **Hint Generation**: NLP-based pipeline utilizing NLTK to intelligently extract contextual clues and generate hints that guide the user towards the correct answer without revealing it outright.
4. **Analytics Dashboard**: A robust Streamlit web interface (`UI/app.py`) for users to test the models interactively, generate quizzes on the fly, and visualize model performance (Accuracy, F1, Precision, Recall, and Confusion Matrices).
5. **Exploratory Data Analysis (EDA)**: Comprehensive `EDA.py` script that explores text length distributions, question types, outliers, and feature correlations.

---

## 📂 Project Structure

```text
├── data/
│   └── transformed/          # Preprocessed datasets (train.csv)
├── docs/
│   └── eda/                  # EDA Visualizations (Boxplots, Heatmaps, etc.)
├── models/                   # Serialized trained models (Joblib format)
│   ├── model_a_lr.pkl
│   ├── model_a_svm.pkl
│   ├── model_a_kmeans.pkl
│   └── ...
├── notebooks/
│   └── EDA.py                # Exploratory Data Analysis script
├── src/
│   ├── model_a_train.py      # Training script for Answer Verification
│   ├── model_b_train.py      # Training script for Distractor Generation
│   ├── inference.py          # Command-line inference and prediction tools
│   ├── evaluate.py           # Model evaluation (Accuracy, BLEU, ROUGE, etc.)
│   └── hint_generator.py     # NLTK-based hint generation engine
├── UI/
│   ├── app.py                # Streamlit Web Application entry point
│   └── components.py         # UI helper functions and model loading logic
├── requirements.txt          # Python package dependencies
└── README.md                 # Project documentation
```

---

## 🚀 Getting Started

### 1. Installation
Clone the repository and install the necessary dependencies:

```bash
# Create and activate a virtual environment (optional but recommended)
python -m venv venv
venv\Scripts\activate  # On Windows

# Install required packages
pip install -r requirements.txt

# Download required NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### 2. Running the Web Dashboard
Launch the interactive Streamlit interface to test the AI Quiz Engine:

```bash
streamlit run UI/app.py
```

### 3. Generating EDA Reports
To generate statistical overviews and visualizations of your dataset:

```bash
python notebooks/EDA.py
```
This will populate the `docs/eda/` folder with correlation heatmaps, outlier boxplots, and distribution charts.

---

## 🧪 Model Training & Evaluation

To train or re-train the underlying models, utilize the scripts in the `src/` directory.

- **Train Answer Verification:** `python src/model_a_train.py`
- **Train Distractor Generator:** `python src/model_b_train.py`
- **Run Evaluation Metrics:** `python src/evaluate.py`

### Supported Evaluation Metrics:
- **Classification:** Accuracy, Precision, Recall, F1-Score (Macro/Micro)
- **Generation:** BLEU, ROUGE, METEOR, Precision@K

---

## 🛠️ Built With
* [Streamlit](https://streamlit.io/) - Web UI & Dashboard
* [Scikit-Learn](https://scikit-learn.org/) - Classical ML (LR, SVM, K-Means)
* [Pandas](https://pandas.pydata.org/) & [NumPy](https://numpy.org/) - Data Manipulation
* [Plotly](https://plotly.com/python/) & [Matplotlib](https://matplotlib.org/) - Visualizations
* [NLTK](https://www.nltk.org/) - Natural Language Processing
