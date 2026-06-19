import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -----------------------------------------------------------
# Step 1: Load data
# -----------------------------------------------------------
df = pd.read_csv(os.path.join(BASE_DIR, "Churn_Modelling.csv"))

# Show columns so we always know what we're working with
print("Columns:", df.columns.tolist())

# -----------------------------------------------------------
# Step 2: Clean up
# -----------------------------------------------------------
# Drop identifier columns that don't help prediction
drop_cols = ['RowNumber', 'CustomerId', 'Surname', 'customerID']
df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

# Encode categorical columns
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

# -----------------------------------------------------------
# Step 3: Set target — handles both common column names
# -----------------------------------------------------------
if 'Exited' in df.columns:
    target = 'Exited'
elif 'Churn' in df.columns:
    target = 'Churn'
else:
    raise ValueError(f"No churn column found. Available columns: {df.columns.tolist()}")

print(f"Target column : {target}")

X = df.drop(target, axis=1)
y = df[target]

# -----------------------------------------------------------
# Step 4: Split & train
# -----------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc  = accuracy_score(y_test,  model.predict(X_test))

print(f"Train Accuracy : {train_acc:.2%}")
print(f"Test  Accuracy : {test_acc:.2%}")

# -----------------------------------------------------------
# Step 5: Two plots
# -----------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Customer Churn Prediction', fontsize=14, fontweight='bold')

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
            xticklabels=['Stayed', 'Churned'],
            yticklabels=['Stayed', 'Churned'])
axes[1].set_title('Confusion Matrix')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'churn_report.png'), dpi=150, bbox_inches='tight')
print("Plot saved to: churn_report.png")
plt.show()