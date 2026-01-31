import pandas as pd
import numpy as np
import os

# Load kaggle dataset
kaggle_path = "data/datasets/raw/train-00000-of-00001.parquet"
df_kaggle = pd.read_parquet(kaggle_path)

# Add Emotions column for more clarity
label_to_emotions = {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise"}
df_kaggle["emotions"] = df_kaggle["label"].map(label_to_emotions)


# Load semeval dataset
sem_eval_path_train = "data/datasets/raw/2018-E-c-En-train.txt"
df_semeval = pd.read_csv(sem_eval_path_train, sep='\t')

# Make sem_eval dataset compatible

# Returns rows that have none of target emotions for the kaggle dataset (all 0s)
def filter_rows(target_df, emotion_mapping):

    # Extract the emotion names from your dictionary
    target_emotions = list(emotion_mapping.values())

    # Filter only the columns that actually exist in the dataframe to avoid KeyErrors
    existing_cols = [col for col in target_emotions if col in target_df.columns]

    # Sum the emotion columns value horizontally (axis=1) and keep rows where the sum is 0
    mask = target_df[existing_cols].sum(axis=1) == 0
    negative_df = target_df[mask].copy()

    print(f"Filtered {len(negative_df)} rows out of {len(target_df)}")
    return negative_df

# Dataset containing only rows with label not present in dataset1
negative = filter_rows(df_semeval, label_to_emotions) 

# Take only compatible rows:
positive = df_semeval[~df_semeval.index.isin(negative.index)]

# Identify which columns in 'positive' map to our target emotions
alignment_map = {
'anger': 'anger',
'fear': 'fear',
'joy': 'joy',
'love': 'love',
'sadness': 'sadness',
'surprise': 'surprise',
'disgust': 'anger' # Common academic grouping: Disgust often maps to Anger
}

# Take only columns i.e labels present in both datasets
present_cols = [col for col in alignment_map.keys() if col in positive.columns]

# Create semeval_subset directly from positive text
semeval_subset = pd.DataFrame({
    'text': positive['Tweet'].values
}, index=positive.index)

# Fill semeval_subset with values from positive and take only the first label (instead of multiple labels)
semeval_subset['emotions'] = positive[present_cols].any(axis=1).astype(int)
def map_to_primary(row):
    for col in present_cols:
        if row[col] == 1:
            return alignment_map[col]

semeval_subset['emotions'] = positive.apply(map_to_primary, axis=1)

# Semeval dataset comes without numerical labels
emotions_to_label = {"sadness": 0, "joy": 1, "love": 2, "anger": 3, "fear": 4, "surprise": 5 }
semeval_subset["label"] = semeval_subset["emotions"].map(emotions_to_label)

# Merge DF kaggle and Semeval
merged_df = pd.concat([df_kaggle, semeval_subset], ignore_index=True)

# Load ELTEA17 Dataset
eltea_dataset_path = "data/datasets/raw/eltea_train.txt"
eltea_df = pd.read_csv(eltea_dataset_path, 
    sep='|', 
    header=None, 
    names=['emotions', 'sarcasm', 'text']
)

# Keep only tweets WITHOUT sarcasm and then drop the column
eltea_df = eltea_df[eltea_df['sarcasm'] =='N']
eltea_df = eltea_df.drop(columns=['sarcasm'])

# Convert emotions to the correct format
conversion_map_eltea = {"joy" : "joy", "sad" : "sadness", "dis" : "anger", "ang" : "anger", "fea": "fear", "sup" : "surprise"}

eltea_df["emotions"] = eltea_df["emotions"].map(conversion_map_eltea)

# Add numeric label to dataset
emotions_to_label = {"sadness": 0, "joy": 1, "love": 2, "anger": 3, "fear": 4, "surprise": 5 }
eltea_df["label"] = eltea_df["emotions"].map(emotions_to_label)

# Merge ELTEA17 and the merged dataset
merged_df = pd.concat([merged_df, semeval_subset])

print(merged_df.head())

# Save to path
project_root = os.getcwd()
save_path = os.path.join(project_root, "data", "datasets", "process")

os.makedirs(save_path, exist_ok=True)
parquet_path = os.path.join(save_path, "merged_emotions.parquet")

merged_df.to_parquet(parquet_path, index=False)

print(f"Database successfully saved under {parquet_path}")


