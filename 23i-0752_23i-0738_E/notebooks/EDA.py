import os
import pandas as pd
import matplotlib.pyplot as plt

EDA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'docs', 'eda')

def run_eda(train_df):
    """
    Generate EDA visualizations and summary statistics.
    All plots are saved to the new project Drive folder.
    """
    os.makedirs(EDA_DIR, exist_ok=True)
    print("\n" + "=" * 55)
    print("  EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 55)

    # -- Data Preparation for Transformed Format --
    if 'is_correct' in train_df.columns and 'answer' not in train_df.columns:
        print("\n  [Info] Restructuring transformed data for EDA...")
        def get_row_data(g):
            g = g.reset_index(drop=True)
            correct_idx = g['is_correct'].idxmax() if g['is_correct'].any() else 0
            res = {'answer': ['A', 'B', 'C', 'D'][correct_idx] if correct_idx < 4 else 'A'}
            for i, label in enumerate(['A', 'B', 'C', 'D']):
                res[label] = g.iloc[i]['option'] if i < len(g) else ''
            return pd.Series(res)
        train_df = train_df.groupby(['article', 'question']).apply(get_row_data, include_groups=False).reset_index()

    if 'word_count' not in train_df.columns:
        train_df['word_count'] = train_df['article'].apply(lambda x: len(str(x).split()))
    if 'char_count' not in train_df.columns:
        train_df['char_count'] = train_df['article'].apply(lambda x: len(str(x)))
    if 'difficulty' not in train_df.columns:
        train_df['difficulty'] = 'unknown'

    # -- 1. Summary Statistics Table --──────────────────────
    print("\n-- Summary Statistics --")
    summary = {
        'Split':             ['Train'],
        'Total Questions':   [len(train_df)],
        'Unique Articles':   [train_df['article'].nunique()],
        'Avg Passage Words': [round(train_df['word_count'].mean(), 1)],
        'Avg Passage Chars': [round(train_df['char_count'].mean(), 1)],
        'Middle Level':      [(train_df['difficulty'] == 'middle').sum()],
        'High Level':        [(train_df['difficulty'] == 'high').sum()],
    }
    summary_df = pd.DataFrame(summary)
    print(summary_df.to_string(index=False))
    summary_df.to_csv(os.path.join(EDA_DIR, 'summary_statistics.csv'), index=False)

    # -- 2. Answer Balance Distribution --──────────────────
    print("\n-- Answer Balance (Train) --")
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

    # -- 3. Passage Length Distribution --───────────────────
    print("\n-- Passage Length Distribution (Train) --")
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

    # -- 4. Difficulty Level Breakdown --────────────────────
    print("\n-- Difficulty Level Breakdown (Train) --")
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
    
    if not middle_wc.empty and not high_wc.empty:
        bp = axes[1].boxplot([middle_wc, high_wc], labels=['Middle School', 'High School'],
                              patch_artist=True)
        bp['boxes'][0].set_facecolor('#4CAF50')
        bp['boxes'][1].set_facecolor('#E91E63')
    else:
        axes[1].text(0.5, 0.5, 'No difficulty data', ha='center', va='center')
        axes[1].axis('off')
        
    axes[1].set_title('Passage Length by Difficulty')
    axes[1].set_ylabel('Word Count')

    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, 'difficulty_breakdown.png'), dpi=150)
    plt.close()
    print(f"  Saved: {EDA_DIR}/difficulty_breakdown.png")

    # -- 5. Question Type Analysis --────────────────────────
    print("\n-- Question Type Analysis (Train) --")

    def classify_question_type(q):
        q_lower = q.lower().strip()
        if q_lower.startswith('what'):   return 'What'
        if q_lower.startswith('which'):  return 'Which'
        if q_lower.startswith('who'):    return 'Who'
        if q_lower.startswith('where'):  return 'Where'
        if q_lower.startswith('when'):   return 'When'
        if q_lower.startswith('why'):    return 'Why'
        if q_lower.startswith('how'):    return 'How'
        # Check for fill-in-the-blank style
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

    # -- 6. Answer Balance by Difficulty --──────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Answer Balance by Difficulty Level (Train)', fontsize=14, fontweight='bold')

    for i, level in enumerate(['middle', 'high']):
        subset = train_df[train_df['difficulty'] == level]
        axes[i].set_title(f'{level.title()} School (n={len(subset)})')
        if subset.empty:
            axes[i].text(0.5, 0.5, 'No data', ha='center', va='center')
            axes[i].axis('off')
            continue
            
        ans_counts = subset['answer'].value_counts().sort_index()
        axes[i].bar(ans_counts.index, ans_counts.values, color=colors, edgecolor='black', linewidth=0.5)
        axes[i].set_xlabel('Answer Choice')
        axes[i].set_ylabel('Count')
        for idx_bar, (label, count) in enumerate(zip(ans_counts.index, ans_counts.values)):
            axes[i].text(idx_bar, count + 100, f'{count}', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, 'answer_by_difficulty.png'), dpi=150)
    plt.close()
    print(f"  Saved: {EDA_DIR}/answer_by_difficulty.png")

    # -- 7. Option Length Analysis --────────────────────────
    print("\n-- Option Length Statistics (Train) --")
    for col in ['A', 'B', 'C', 'D']:
        if col in train_df.columns:
            avg_len = train_df[col].apply(lambda x: len(str(x).split())).mean()
            print(f"  Avg words in Option {col}: {avg_len:.1f}")

    # -- 8. Data Overview & Missing Values --────────────────
    print("\n-- Data Overview & Missing Values (Train) --")
    missing_data = train_df.isnull().sum()
    missing_percent = (missing_data / len(train_df)) * 100
    missing_df = pd.DataFrame({'Missing Values': missing_data, 'Percentage': missing_percent})
    print(missing_df.to_string())
    missing_df.to_csv(os.path.join(EDA_DIR, 'missing_values_analysis.csv'))

    # -- 9. Outlier Detection --─────────────────────────────
    print("\n-- Outlier Detection (Train) --")
    numeric_cols = train_df.select_dtypes(include=['int64', 'float64']).columns
    outliers_summary = {}
    
    fig, axes = plt.subplots(1, len(numeric_cols), figsize=(14, 5))
    if len(numeric_cols) == 1:
        axes = [axes]
    fig.suptitle('Outlier Visualization (Boxplots)', fontsize=14, fontweight='bold')
    
    for i, col in enumerate(numeric_cols):
        Q1 = train_df[col].quantile(0.25)
        Q3 = train_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = train_df[(train_df[col] < lower_bound) | (train_df[col] > upper_bound)]
        outliers_summary[col] = len(outliers)
        print(f"  {col}: {len(outliers)} outliers ({(len(outliers)/len(train_df))*100:.1f}%)")
        
        # Visualize outliers with boxplot
        axes[i].boxplot(train_df[col].dropna(), patch_artist=True, boxprops=dict(facecolor='#FF9800'))
        axes[i].set_title(f'{col} Outliers')
        axes[i].set_ylabel('Count')

    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, 'outliers_visualization.png'), dpi=150)
    plt.close()
    print(f"  Saved: {EDA_DIR}/outliers_visualization.png")

    # -- 10. Data Distribution Analysis --───────────────────
    if 'word_count' in train_df.columns and 'char_count' in train_df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Data Distribution Analysis (Violin Plots)', fontsize=14, fontweight='bold')
        
        axes[0].violinplot(train_df['word_count'], showmeans=True, showmedians=True)
        axes[0].set_title('Word Count Distribution')
        axes[0].set_ylabel('Word Count')
        axes[0].set_xticks([1])
        axes[0].set_xticklabels(['Train Data'])

        axes[1].violinplot(train_df['char_count'], showmeans=True, showmedians=True)
        axes[1].set_title('Character Count Distribution')
        axes[1].set_ylabel('Character Count')
        axes[1].set_xticks([1])
        axes[1].set_xticklabels(['Train Data'])

        plt.tight_layout()
        plt.savefig(os.path.join(EDA_DIR, 'data_distribution_analysis.png'), dpi=150)
        plt.close()
        print(f"  Saved: {EDA_DIR}/data_distribution_analysis.png")

    # -- 11. Correlation Analysis --─────────────────────────
    if len(numeric_cols) >= 2:
        print("\n-- Correlation Analysis (Train) --")
        corr_matrix = train_df[numeric_cols].corr()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        cax = ax.matshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        fig.colorbar(cax)
        
        ax.set_xticks(range(len(numeric_cols)))
        ax.set_yticks(range(len(numeric_cols)))
        ax.set_xticklabels(numeric_cols, rotation=45, ha='left')
        ax.set_yticklabels(numeric_cols)
        
        for i in range(len(numeric_cols)):
            for j in range(len(numeric_cols)):
                ax.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}", ha='center', va='center', color='black')

        ax.set_title('Correlation Matrix of Numeric Features', pad=20, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(EDA_DIR, 'correlation_analysis.png'), dpi=150)
        plt.close()
        print(f"  Saved: {EDA_DIR}/correlation_analysis.png")

    # -- 12. Feature Relationship --─────────────────────────
    if 'word_count' in train_df.columns and 'char_count' in train_df.columns and 'difficulty' in train_df.columns:
        print("\n-- Feature Relationship Analysis --")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        middle = train_df[train_df['difficulty'] == 'middle']
        high = train_df[train_df['difficulty'] == 'high']
        
        sample_size = min(2000, len(middle), len(high))
        if sample_size > 0:
            middle_sample = middle.sample(n=sample_size, random_state=42)
            high_sample = high.sample(n=sample_size, random_state=42)
            
            ax.scatter(middle_sample['word_count'], middle_sample['char_count'], alpha=0.5, label='Middle', color='#4CAF50', s=10)
            ax.scatter(high_sample['word_count'], high_sample['char_count'], alpha=0.5, label='High', color='#E91E63', s=10)
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No difficulty data available for this plot.', ha='center')
            ax.axis('off')
            
        ax.set_title('Word Count vs Character Count by Difficulty', fontsize=14, fontweight='bold')
        ax.set_xlabel('Word Count')
        ax.set_ylabel('Character Count')
        
        plt.tight_layout()
        plt.savefig(os.path.join(EDA_DIR, 'feature_relationship.png'), dpi=150)
        plt.close()
        print(f"  Saved: {EDA_DIR}/feature_relationship.png")

    print(f"\n  EDA complete! All plots saved to '{EDA_DIR}/'")
    print("=" * 55)

    return train_df

if __name__ == "__main__":
    # Provide the path to the transformed datasets
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'transformed')
    
    print(f"Loading data from {DATA_DIR}...")
    try:
        train_data = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
        
        # Run the EDA
        run_eda(train_data)
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print("Please make sure train.csv exists in data/transformed/")
