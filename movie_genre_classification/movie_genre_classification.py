import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# -----------------------------------------------------------
# Step 1: Read the data files
# -----------------------------------------------------------

def load_train(filename):
    """Train file has 4 parts: ID ::: Title ::: Genre ::: Plot"""
    filepath = os.path.join(BASE_DIR, filename)
    ids, titles, genres, plots = [], [], [], []

    with open(filepath, encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(' ::: ')
            if len(parts) == 4:
                ids.append(parts[0])
                titles.append(parts[1])
                genres.append(parts[2])
                plots.append(parts[3])

    df = pd.DataFrame({'id': ids, 'title': titles, 'genre': genres, 'plot': plots})
    print(f"Train: {len(df)} movies loaded")
    return df


def load_test(filename):
    """Test file has 3 parts (no genre): ID ::: Title ::: Plot"""
    filepath = os.path.join(BASE_DIR, filename)
    ids, titles, plots = [], [], []

    with open(filepath, encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(' ::: ')
            if len(parts) == 3:
                ids.append(parts[0])
                titles.append(parts[1])
                plots.append(parts[2])
            elif len(parts) == 4:
                # test file also has genre column — handle both cases
                ids.append(parts[0])
                titles.append(parts[1])
                plots.append(parts[3])

    df = pd.DataFrame({'id': ids, 'title': titles, 'plot': plots})
    print(f"Test:  {len(df)} movies loaded")
    return df


train = load_train('train_data.txt')
test  = load_test('test_data.txt')

print(f"Genres: {sorted(train['genre'].unique())}\n")


# -----------------------------------------------------------
# Step 2: Turn plot text into numbers (TF-IDF)
# -----------------------------------------------------------
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)

X_train = vectorizer.fit_transform(train['plot'])
X_test  = vectorizer.transform(test['plot'])


# -----------------------------------------------------------
# Step 3: Train the model
# -----------------------------------------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, train['genre'])


# -----------------------------------------------------------
# Step 4: Predict genres for the test set & save to CSV
# -----------------------------------------------------------
test['predicted_genre'] = model.predict(X_test)

output_path = os.path.join(BASE_DIR, 'predictions.csv')
test[['id', 'title', 'predicted_genre']].to_csv(output_path, index=False)
print(f"Predictions saved to: predictions.csv")
print(test[['id', 'title', 'predicted_genre']].head(10).to_string(index=False))


# -----------------------------------------------------------
# Step 5: Predict the genre of a brand-new plot
# -----------------------------------------------------------
my_plot = ["A detective investigates a mysterious murder in a small town."]
result  = model.predict(vectorizer.transform(my_plot))[0]
print(f"\nCustom plot predicted genre: {result}")