import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix


# -----------------------------------------------------------
# Step 1: Load & prepare data
# -----------------------------------------------------------
df = pd.read_csv("spam.csv", encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'message']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

X = df['message']
y = df['label']

# -----------------------------------------------------------
# Step 2: TF-IDF + Split
# -----------------------------------------------------------
vectorizer = TfidfVectorizer(stop_words='english')
X_tfidf = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42
)

# -----------------------------------------------------------
# Step 3: Train
# -----------------------------------------------------------
model = MultinomialNB()
model.fit(X_train, y_train)

train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc  = accuracy_score(y_test,  model.predict(X_test))

print(f"Train Accuracy : {train_acc:.2%}")
print(f"Test  Accuracy : {test_acc:.2%}")

# -----------------------------------------------------------
# Step 4: Two plots
# -----------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Spam SMS Detection', fontsize=14, fontweight='bold')

# Plot 1 — Train vs Test Accuracy
axes[0].bar(['Train', 'Test'], [train_acc, test_acc],
            color=['steelblue', 'coral'], edgecolor='white', width=0.4)
axes[0].set_title('Train vs Test Accuracy')
axes[0].set_ylabel('Accuracy')
axes[0].set_ylim(0, 1)
for i, v in enumerate([train_acc, test_acc]):
    axes[0].text(i, v + 0.01, f'{v:.2%}', ha='center', fontweight='bold')

# Plot 2 — Confusion Matrix
cm = confusion_matrix(y_test, model.predict(X_test))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1],
            xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
axes[1].set_title('Confusion Matrix')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

plt.tight_layout()
plt.savefig('spam_report.png', dpi=150, bbox_inches='tight')
print("Plot saved to: spam_report.png")
plt.show()

# -----------------------------------------------------------
# Step 5: Custom prediction
# -----------------------------------------------------------
msg = ["Congratulations! You won a free iPhone."]
result = model.predict(vectorizer.transform(msg))[0]
print(f"\nCustom message: {'Spam' if result == 1 else 'Ham'}")