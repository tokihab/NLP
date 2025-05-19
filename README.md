# NLP Repository

## Projects

### 1. Arabic Offensive Comment Detection
**Objective**: Identify offensive content in Arabic social media comments.  
**Dataset**: [ArabicOffensiveComments.xlsx](path/to/dataset) (offensive/non-offensive labels).  

#### Key Components:
- **Preprocessing**: URL/special character removal, Arabic normalization (`pyarabic`), stopword removal, stemming (`tashaphyne`).
- **Features**: Emoji detection, repeated/feminine/negative/question words, long words.
- **Analysis**: Chi-square tests, correlation heatmaps, emoji frequency.

#### Findings:
- Non-offensive comments had marginally higher repetition rates.
- Weak correlation between feminine words and offensive content (p=0.47).
- [Code](NLPTSK.pdf)

---

### 2. IMDB Sentiment Analysis
**Objective**: Classify IMDB movie reviews as positive/negative using classical ML.  
**Dataset**: [50,000 labeled reviews](https://ai.stanford.edu/~amaas/data/sentiment/).  

#### Workflow:
- **Preprocessing**: HTML tag removal, lowercasing, lemmatization, TF-IDF (top 20k features).
- **Models**: Logistic Regression (88%), SVM (89.5%), Multinomial Naive Bayes (85%).
- **Evaluation**: Confusion matrices, precision/recall/F1, hyperparameter tuning (5-fold CV).
- **GUI**: Tkinter interface for predictions.
