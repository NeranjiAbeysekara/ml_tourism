import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("cleaned_accommodation_data.csv")

# 1. Distribution of Accommodation Types
plt.figure(figsize=(12, 6))
sns.countplot(y='Type', data=df, order=df['Type'].value_counts().index, palette='viridis')
plt.title('Distribution of Accommodation Types')
plt.xlabel('Count')
plt.ylabel('Type')
plt.tight_layout()
plt.savefig('cleaned_type_distribution.png')

# 2. Distribution of Rooms (Boxplot to see outliers)
plt.figure(figsize=(10, 5))
sns.boxplot(x='Rooms', data=df)
plt.title('Distribution of Room Counts')
plt.tight_layout()
plt.savefig('rooms_distribution_boxplot.png')

# 3. Top Districts with most accommodations
plt.figure(figsize=(12, 6))
df['District'].value_counts().head(10).plot(kind='bar', color='skyblue')
plt.title('Top 10 Districts by Number of Accommodations')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('top_districts.png')

print("Visualizations saved: 'cleaned_type_distribution.png', 'rooms_distribution_boxplot.png', 'top_districts.png'")

# Summary statistics
print("\nSummary of Rooms:")
print(df['Rooms'].describe())

print("\nDistrict-wise Accommodation Count (Top 5):")
print(df['District'].value_counts().head())
