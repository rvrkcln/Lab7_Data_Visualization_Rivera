import pandas as pd 
import os 

# Define the dataset path using your absolute file path 
dataset_path = "spotify_top_1000_tracks.csv"

# Load dataset 
df = pd.read_csv(dataset_path, encoding="utf-8") 

# Convert release_date and extract year 
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce') 
df['year'] = df['release_date'].dt.year 

# FIX: We REMOVE the line that tried to create 'duration_min'  
# because it already exists in the CSV file you loaded. 
# (The 'duration_min' column is ready for use!) 
print("Dataset loaded and basic preprocessing complete!") 
print(df.head(3))

import numpy as np 

# Clean up text columns 
df['track_name'] = df['track_name'].str.strip() 
df['artist'] = df['artist'].str.strip() 
df['album'] = df['album'].str.strip() 

# Convert 'year' to integer 
df['year'] = df['year'].fillna(0).astype(int)

# Drop unnecessary columns 
cols_to_drop = ['spotify_url', 'id'] 
 
# Check for and add other common audio feature columns if they exist 
if 'time_signature' in df.columns: 
    cols_to_drop.append('time_signature') 
if 'key' in df.columns: 
    cols_to_drop.append('key') 
if 'mode' in df.columns: 
    cols_to_drop.append('mode') 
 
df = df.drop(columns=cols_to_drop, errors='ignore') 
 
# Feature Engineering: Tempo Category 
tempo_bins = [0, 100, 140, np.inf] 
tempo_labels = ['Slow', 'Medium', 'Fast'] 
 
if 'tempo' in df.columns: 
    df['tempo_category'] = pd.cut(  # Create tempo category column 
        df['tempo'], bins=tempo_bins,  
        labels=tempo_labels, right=False 
    ) 
    print("Feature 'tempo_category' created.") 
else: 
    print("Warning: 'tempo' column not found; skipping 'tempo_category' creation.") 
 
# Remove duplicates 
df = df.drop_duplicates(subset=['track_name', 'artist'], keep='first') 
 
print(f"Data cleaning and feature engineering complete.") 
print(f"Final Row Count after deduplication: {len(df)}") 