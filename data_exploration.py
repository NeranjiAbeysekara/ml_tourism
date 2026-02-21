import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
file_path = "Information for Accommodation.csv"
df = pd.read_csv(file_path)

# Display basic info
print("Dataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nValue counts for 'Type':")
print(df['Type'].value_counts())

print("\nValue counts for 'Grade':")
print(df['Grade'].value_counts())

print("\nValue counts for 'District':")
print(df['District'].value_counts())

# Basic cleaning for exploration
# Drop rows with missing 'Grade' if we want to predict Grade
df_grade = df.dropna(subset=['Grade'])

plt.figure(figsize=(12, 6))
sns.countplot(y='Grade', data=df_grade, order=df_grade['Grade'].value_counts().index)
plt.title('Distribution of Accommodation Grades')
plt.savefig('grade_distribution.png')
print("\nSaved grade_distribution.png")

plt.figure(figsize=(10, 8))
sns.countplot(y='Type', data=df)
plt.title('Distribution of Accommodation Types')
plt.savefig('type_distribution.png')
print("Saved type_distribution.png")
