# 🤖 Machine Learning Projects

> A curated collection of end-to-end machine learning projects covering classification, NLP, and predictive analytics — from raw data to trained models and visual reports.

---

## 📁 Repository Structure

```
├── customer_churn_prediction/
│   ├── Churn_Modelling.csv
│   ├── customer_churn_prediction.py
│   └── churn_report.png
│
├── movie_genre_classification/
│   ├── movie_genre_classification.py
│   ├── train_data.txt
│   ├── test_data.txt
│   └── predictions.csv
│
└── sms_spam_detection/
    ├── sms_spam_detection.py
    ├── spam.csv
    └── spam_report.png
```

---

## 📌 Projects

### 1. 🔄 Customer Churn Prediction

**Goal:** Predict whether a bank customer will churn (leave) based on demographic and account data.

| Detail | Info |
|--------|------|
| **Dataset** | `Churn_Modelling.csv` |
| **Task** | Binary Classification |
| **Output** | Churn probability per customer + visual report |

**Highlights:**
- Feature engineering on customer demographics, account balance, and activity metrics
- Trained classification model with evaluation metrics (accuracy, precision, recall, F1)
- Visual summary exported to `churn_report.png`

---

### 2. 🎬 Movie Genre Classification

**Goal:** Automatically classify movies into genres based on plot descriptions using NLP.

| Detail | Info |
|--------|------|
| **Dataset** | `train_data.txt` / `test_data.txt` |
| **Task** | Multi-class Text Classification |
| **Output** | Genre predictions saved to `predictions.csv` |

**Highlights:**
- Text preprocessing pipeline (tokenization, stopword removal, vectorization)
- Multi-class classification using machine learning on plot summaries
- Predictions exported in structured CSV format for downstream use

---

### 3. 📩 SMS Spam Detection

**Goal:** Detect whether an SMS message is spam or legitimate (ham) using NLP techniques.

| Detail | Info |
|--------|------|
| **Dataset** | `spam.csv` |
| **Task** | Binary Text Classification |
| **Output** | Spam classification model + visual report |

**Highlights:**
- Bag-of-Words / TF-IDF text vectorization
- Trained spam classifier with precision/recall optimized for low false positives
- Visual performance summary exported to `spam_report.png`

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)

- **Language:** Python 3.8+
- **ML Framework:** scikit-learn
- **Data Handling:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **NLP:** NLTK / scikit-learn text utilities

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy scikit-learn matplotlib seaborn nltk
```

### Run a Project

```bash
# Customer Churn Prediction
python customer_churn_prediction/customer_churn_prediction.py

# Movie Genre Classification
python movie_genre_classification/movie_genre_classification.py

# SMS Spam Detection
python sms_spam_detection/sms_spam_detection.py
```

---

## 📊 Results at a Glance

| Project | Task | Key Metric |
|--------|------|------------|
| Customer Churn Prediction | Binary Classification | Accuracy / AUC-ROC |
| Movie Genre Classification | Multi-class NLP | F1-Score (macro) |
| SMS Spam Detection | Binary NLP | Precision / Recall |

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome. Feel free to open a pull request or file an issue.

---

## 📄 License

This repository is open-source and available under the [MIT License](LICENSE).
