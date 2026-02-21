import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("cleaned_accommodation_data.csv")

print("--- Grade Distribution Analysis ---")
grade_counts = df['Grade'].value_counts()
print(grade_counts)

# Calculate percentage
grade_pct = df['Grade'].value_counts(normalize=True) * 100
print("\nPercentage Distribution:")
print(grade_pct)

# Save a specific plot for the report showing imbalance
plt.figure(figsize=(10, 6))
grade_counts.plot(kind='bar', color='salmon')
plt.title('Distribution of Target Variable (Grade)')
plt.ylabel('Number of Accommodations')
plt.xlabel('Grade')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('grade_imbalance_analysis.png')

print("\nSaved 'grade_imbalance_analysis.png' for the report.")

# Observation:
# If some grades have < 10 samples, the model will struggle.
# 'C' might only have a few, 'FIVE' stars might be rare.
