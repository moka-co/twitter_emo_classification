from src.VocabBuilder import VocabBuilder
import pandas as pd
import os

save_filepath = "data/vocab.json"
load_filepath = "data/datasets/process/outliers_removed.parquet"

dictbuild = VocabBuilder()

df = pd.read_parquet(load_filepath)

# Build the dictionary
dictbuild.build(df['processed_text'])

df['sequences'] = df['processed_text'].apply(dictbuild.transform)

dictbuild.save(save_filepath)

save_final_df = "data/datasets/final/"
os.makedirs(save_final_df, exist_ok=True)
parquet_path= save_final_df + "dataset.parquet"

df.to_parquet(parquet_path, index=False, engine='pyarrow')

