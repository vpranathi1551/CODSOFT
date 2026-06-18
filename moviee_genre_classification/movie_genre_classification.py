# Movie Genre Classification

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("movies.csv")

# Example columns:
# plot, genre

X = df['plot']
y = df['genre']

# Convert text to features
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=5000
)

X_tfidf = vectorizer.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y,
    test_size=0.2,
    random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

# Example prediction
new_plot = [
    "A detective investigates a mysterious murder in a small town."
]

new_plot_tfidf = vectorizer.transform(new_plot)

print("Predicted Genre:",
    model.predict(new_plot_tfidf)[0])