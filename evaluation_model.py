import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import seaborn as sns
import matplotlib.pyplot as plt

# === 1. Load Data ===
vectorised_file = 'vectorised.csv'
df = pd.read_csv(vectorised_file)
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

X = df.drop(columns=['status'])
y = df['status']

# === 2. Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 3. Define Classifiers ===
models = {
    'SVM': SVC(C=10, kernel='rbf', class_weight='balanced'),
    'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=1000),
    'Random Forest': RandomForestClassifier(class_weight='balanced', random_state=42)
}

# === 4. Evaluate Each Model ===
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n=== {name} ===")
    print(classification_report(y_test, y_pred))
    print("Accuracy:", acc)

# === 5. Confusion Matrix for Best Model (e.g., Random Forest) ===
best_model = models['Random Forest']
y_pred_best = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Random Forest')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()

# === 6. Predict on Full Data (Optional) ===
y_pred_full = best_model.predict(X)
pred_df = pd.DataFrame({'prediction': y_pred_full})
pred_df.to_csv('predictions.csv', index=False)
print("\nPredictions saved to predictions.csv")
