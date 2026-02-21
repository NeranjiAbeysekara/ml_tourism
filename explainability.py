import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance, PartialDependenceDisplay

# 1. Load the Model and Data
package = joblib.load('tourism_model_package.pkl')
model = package['model']
encoder = package['encoder']

df = pd.read_csv("cleaned_accommodation_data.csv")
features = ['Type', 'Rooms', 'District', 'Longitude', 'Latitude']
X = df[features].copy()
X[['Type', 'District']] = encoder.transform(X[['Type', 'District']]).astype(int)
y = df['Unified_Grade']

print("Generating Explainability Analysis...")

# 2. Feature Importance (Permutation based)
# This shows how much each feature affects the accuracy
result = permutation_importance(model, X, y, n_repeats=10, random_state=42, n_jobs=-1)
sorted_idx = result.importances_mean.argsort()

plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_idx)), result.importances_mean[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), [features[i] for i in sorted_idx])
plt.title("Feature Importance (Permutation Importance)")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig('report_images/feature_importance.png')
print("Saved 'report_images/feature_importance.png'")

# 3. Partial Dependence Plots (PDP)
# This shows how the target prediction changes as a specific feature (like Rooms) changes
fig, ax = plt.subplots(figsize=(12, 8))
display = PartialDependenceDisplay.from_estimator(
    model, X, features=['Rooms', 'Latitude', 'Longitude'],
    target='Luxury', # Show likelihood of being Luxury
    ax=ax
)
plt.suptitle('Partial Dependence of "Luxury" Grade')
plt.tight_layout()
plt.savefig('report_images/pdp_analysis.png')
print("Saved 'report_images/pdp_analysis.png'")

print("\nExplainability report generated successfully.")
