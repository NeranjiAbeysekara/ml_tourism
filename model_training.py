import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from catboost import CatBoostClassifier
from sklearn.preprocessing import OrdinalEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# 1. Load the preprocessed data
df = pd.read_csv("cleaned_accommodation_data.csv")

# 2. Select Features and Target
features = ['Type', 'Rooms', 'District', 'Longitude', 'Latitude']
target = 'Unified_Grade'

X = df[features].copy()
y = df[target].copy()

# Categorical features list
categorical_features = ['Type', 'District']

# Encode categorical variables as integers
# CatBoost can handle strings, but using an encoder helps maintain 
# consistency for the Streamlit app's existing logic.
encoder = OrdinalEncoder()
X[categorical_features] = encoder.fit_transform(X[categorical_features]).astype(int)

# 3. Train/Test Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

print(f"Training set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

# 4. Initialize CatBoostClassifier
# Specify categorical_features indices: [0, 2] corresponding to 'Type' and 'District'
model = CatBoostClassifier(
    iterations=200,
    learning_rate=0.1,
    depth=6,
    cat_features=[0, 2],
    random_seed=42,
    verbose=0  # Silent training
)

# 5. Train the Model
print("\nStarting CatBoost training...")
model.fit(X_train, y_train)

# 6. Evaluation
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"\nEvaluation Results (CatBoost):")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1-Score: {f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 7. Confusion Matrix Plot
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=model.classes_, yticklabels=model.classes_)
plt.title('Confusion Matrix - CatBoost')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('report_images/confusion_matrix.png')
print("\nSaved 'report_images/confusion_matrix.png'")

# 8. Save the Model and Encoder
joblib.dump({'model': model, 'encoder': encoder}, 'tourism_model_package.pkl')
print("Model and Encoder saved as 'tourism_model_package.pkl'")
