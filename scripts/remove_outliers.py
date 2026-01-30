import pandas as pd
import numpy as np
import time
from src.PreprocessPipeline import PreprocessPipeline

# Load dataset
dataset_path = "data/datasets/process/merged_emotions.parquet"

df = pd.read_parquet(dataset_path)

# Initialize Pipeline
pipeline = PreprocessPipeline()

print(f"Applying Preprocessing Pipeline...")
start_time = time.time()
df['processed_text'] = df['text'].apply(lambda x: pipeline.transform(str(x)))
df['token_count'] = df['processed_text'].apply(len)

end_time = time.time()
print(f"Success! Processed {len(df)} rows in {(end_time - start_time):.2f}s")

# Compute stats based on tokens, not characters
token_counts = df['token_count'].values
mean_len = np.mean(token_counts)
std_len = np.std(token_counts)


# Calculate the delimiter: Mean + 3*Sigma
outliers_del = int(mean_len + (3 * std_len))

print(f"\n--- Statistics ---")
print(f"Token Mean: {mean_len:.2f}")
print(f"Token Std:  {std_len:.2f}")
print(f"Outlier Delimiter (3-sigma): {outliers_del} tokens")

# Filter
print(f"\n--- Filtering ---")
print(f"Original length: {len(df)}")

filtered_df = df[df['token_count'] <= outliers_del].copy()

deleted_count = len(df) - len(filtered_df)
print(f"Filtered length: {len(filtered_df)}")
print(f"Deleted samples: {deleted_count}")

parquet_path = "data/datasets/process/outliers_removed.parquet"
filtered_df.to_parquet(parquet_path, index=False, engine='pyarrow')
print("Database successfully saved")
