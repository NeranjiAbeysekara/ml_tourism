import pandas as pd
import numpy as np

def clean_and_preprocess(file_path):
    # Load dataset
    df = pd.read_csv(file_path)
    
    # 1. Column Rename
    df.rename(columns={'Logitiute': 'Longitude'}, inplace=True)
    
    # 2. Handle 'Rooms' column
    df['Rooms'] = pd.to_numeric(df['Rooms'], errors='coerce')
    df['Rooms'] = df['Rooms'].fillna(df['Rooms'].median())
    
    # 3. Handle 'Latitude' and 'Longitude'
    df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
    df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
    
    # Fill missing coordinates with District medians
    df['Latitude'] = df.groupby('District')['Latitude'].transform(lambda x: x.fillna(x.median()))
    df['Longitude'] = df.groupby('District')['Longitude'].transform(lambda x: x.fillna(x.median()))
    
    # If still NaN, fill with overall median
    df['Latitude'] = df['Latitude'].fillna(df['Latitude'].median())
    df['Longitude'] = df['Longitude'].fillna(df['Longitude'].median())
    
    # 4. Handle 'Grade' consistency
    df['Grade'] = df['Grade'].fillna('UNCLASSIFIED').str.strip().str.upper()
    
    # 5. FEATURE ENGINEERING: Unified Grade
    # Luxury: FIVE, FOUR, DELUXE, A
    # Mid: THREE, TWO, SUPERIOR, B
    # Budget: ONE, STANDARD, C, UNCLASSIFIED
    luxury_labels = ['FIVE', 'FOUR', 'DELUXE', 'A']
    mid_labels = ['THREE', 'TWO', 'SUPERIOR', 'B']
    
    def unify_grade(grade):
        if grade in luxury_labels:
            return 'Luxury'
        elif grade in mid_labels:
            return 'Mid-Range'
        else:
            return 'Budget'
            
    df['Unified_Grade'] = df['Grade'].apply(unify_grade)
    
    # 6. Categorical Cleanup
    categorical_cols = ['Type', 'District', 'AGA Division', 'PS/MC/UC']
    for col in categorical_cols:
        df[col] = df[col].fillna('Unknown').str.strip()
        
    # Save the cleaned data
    df.to_csv("cleaned_accommodation_data.csv", index=False)
    print("Preprocessed & Engineered data saved to 'cleaned_accommodation_data.csv'")
    
    return df

if __name__ == "__main__":
    file_path = "Information for Accommodation.csv"
    processed_data = clean_and_preprocess(file_path)
    
    print("\nValue counts for Unified_Grade:")
    print(processed_data['Unified_Grade'].value_counts())
    print("\nSample of engineered data:")
    print(processed_data[['Name', 'Grade', 'Unified_Grade']].head(10))
