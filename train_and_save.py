# train_and_save.py - Train multiple ML models for disease prediction and save them

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
print("\n🔄 Loading dataset...")
data = pd.read_csv("Training.csv")

# Encode target labels
y = LabelEncoder()
data['prognosis'] = y.fit_transform(data['prognosis'])

# Split features and target
X = data.drop(columns=['prognosis'])
y_vals = data['prognosis']

# Save column list for input form in UI
joblib.dump(X.columns.tolist(), "symptoms_list.joblib")
joblib.dump(y, "label_encoder.pkl")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_vals, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, 'scaler.joblib')

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
    'Naive Bayes': GaussianNB(),
    'Support Vector Machine': SVC(kernel='rbf', probability=True, random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', n_estimators=100, random_state=42)
}

# Train and evaluate models
results = {}
print("\n🧠 Training models...")
for name, model in models.items():
    print(f"\n🔹 {name}")
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {accuracy:.3f}")
    print(f"CV Mean: {cv_scores.mean():.3f}, CV Std: {cv_scores.std():.3f}")

    # Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=False, cmap="Blues")
    plt.title(f"Confusion Matrix - {name}")
    plt.savefig(f"conf_matrix_{name.replace(' ', '_').lower()}.png")
    plt.close()

    # Store result
    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }

    # Save model
    joblib.dump(model, f"{name.replace(' ', '_').lower()}_model.joblib")

# Determine best model by test accuracy
best_model_name = max(results.items(), key=lambda x: x[1]['accuracy'])[0]
print(f"\n✅ Best Model: {best_model_name} with accuracy {results[best_model_name]['accuracy']:.3f}")
joblib.dump(best_model_name, "best_model_name.joblib")

# Print model comparison
print("\n📊 Model Comparison:")
print(f"{'Model':<25}{'Accuracy':<10}{'CV Mean':<10}{'CV Std':<10}")
print("-" * 55)
for name, res in results.items():
    print(f"{name:<25}{res['accuracy']:.3f}{res['cv_mean']:.3f}{res['cv_std']:.3f}")
print("\n🎉 All models trained and saved successfully!")