# 🌍 Multilingual App Reviews Sentiment Analysis

![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Machine Learning](https://img.shields.io/badge/ML-85.2%25%20Accuracy-brightgreen)
![Languages](https://img.shields.io/badge/Languages-5%20Supported-orange)

A beginner-friendly machine learning project that demonstrates realistic sentiment analysis across multiple languages using TF-IDF and Logistic Regression.

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Results](#-results)
- [How to Run](#-how-to-run)
- [Technical Details](#-technical-details)
- [License](#-license)

## 📋 Project Overview

This project demonstrates practical sentiment analysis on multilingual app reviews using standard machine learning techniques. The goal is to provide a **realistic educational example** that shows actual ML challenges and achievable performance, rather than unrealistic perfect results.

**Key Learning Points:**
- Real-world datasets have noise and complexity
- ML models achieve good but not perfect accuracy
- Cross-language sentiment analysis is challenging
- Proper evaluation includes error analysis

## 📊 Dataset

### Source
- **Type**: Synthetic multilingual app reviews (educational purpose)
- **Generation**: Programmatically created with realistic ML challenges
- **License**: MIT (free for educational and commercial use)

### Characteristics
| Metric | Value |
|--------|-------|
| **Total Reviews** | 1,112 samples |
| **Languages** | 5 (English, Spanish, French, German, Italian) |
| **Sentiment Distribution** | 55.3% Negative, 44.7% Positive (realistic imbalance) |
| **Complexity Types** | Clear sentiment (59.7%), Mixed sentiment (20.6%), Ambiguous (14.3%), Very short (5.4%) |
| **Data Quality Challenges** | Label noise (5%), Class imbalance, Varying text lengths |

### Realistic ML Challenges
- **Mixed sentiment** reviews with both positive and negative words
- **Ambiguous** reviews with neutral language
- **Class imbalance** reflecting real-world data
- **Label noise** simulating annotation errors
- **Short texts** with limited context

## 📈 Results

### Overall Performance
| Metric | Baseline (Logistic Regression) |
|--------|-------------------------------|
| **Accuracy** | **85.2%** |
| **F1-Score** | **83.1%** |
| **Precision** | 85.0% (weighted avg) |
| **Recall** | 85.0% (weighted avg) |

### Per-Language Performance
| Language | Accuracy | F1-Score | Test Samples |
|----------|----------|----------|--------------|
| **English** | **91.5%** | **90.9%** | 47 |
| **German** | **86.7%** | **80.0%** | 45 |
| **Italian** | **83.3%** | **85.7%** | 42 |
| **Spanish** | **82.9%** | **74.1%** | 41 |
| **French** | **81.2%** | **80.0%** | 48 |

### Key Insights
- ✅ **Realistic performance** showing actual ML capabilities
- ✅ **Cross-language consistency** with low variance (4.1% std dev)
- ✅ **English performs best** (91.5% accuracy) due to training data
- ⚠️ **Challenging cases** include mixed sentiment and ambiguous reviews
- 📊 **Average confidence**: 73.6% (appropriate uncertainty awareness)

## � How to Run

### Option 1: VS Code with Jupyter Extension (Recommended)
```bash
# 1. Clone the repository
git clone https://github.com/jagjeetjenagit/multilingual-app-reviews-sentiment-analysis.git
cd multilingual-app-reviews-sentiment-analysis

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS/Linux

# 3. Install exact dependencies
pip install -r requirements.txt

# 4. Open in VS Code
code .
# Open multilingual_sentiment_analysis.ipynb
# Run all cells (Ctrl+Shift+P -> "Run All Cells")
```

### Option 2: Traditional Jupyter Notebook
```bash
# Steps 1-3 same as above, then:
jupyter notebook
# Navigate to multilingual_sentiment_analysis.ipynb
# Run all cells
```

### Quick Test
```python
# Test the trained model (run this in the final cell)
predict_sentiment("This app is amazing!")
# Expected: {'sentiment': 'positive', 'confidence': 0.85, 'emoji': '😊'}
```

## 🔧 Technical Details

### Architecture
- **Algorithm**: TF-IDF + Logistic Regression with balanced class weights
- **Features**: 5,000 most informative TF-IDF features (unigrams + bigrams)
- **Preprocessing**: Text normalization, stopword removal, stratified sampling
- **Validation**: Stratified train/test split (80/20) with reproducible seeds

### Model Configuration
```python
# Key parameters achieving 85.2% accuracy
TfidfVectorizer(max_features=5000, ngram_range=(1,2), min_df=2, max_df=0.95)
LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
```

### Why This Performance Is Good
- **Realistic baseline** for multilingual sentiment analysis
- **Handles noise** including 5% label errors and class imbalance
- **Cross-language consistency** with minimal bias (4.1% accuracy std dev)
- **Educational value** showing real ML challenges and solutions

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**MIT License Summary:**
- ✅ Free for educational and commercial use
- ✅ Modify and adapt for your needs  
- ✅ Share and redistribute freely
- ⚖️ Provided "as is" without warranty

---

**🎯 Built with ❤️ using Python, scikit-learn, TF-IDF, and fundamental ML principles.**

**� A realistic educational example showing practical sentiment analysis with achievable results.**

**🌟 Perfect for learning ML fundamentals and understanding real-world challenges!**
