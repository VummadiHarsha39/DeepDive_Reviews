import pandas as pd
from datasets import load_dataset

def download_goemotions(split='train'):
    """
    Downloads the GoEmotions dataset and converts it to a pandas DataFrame.
    Simplifies multi-label emotions into one-hot encoded columns.
    """
    print(f"Downloading GoEmotions dataset (split: {split})...")
    # Load the 'simplified' version which has 27 emotions + 'no_emotion'
    dataset = load_dataset("go_emotions", "simplified")[split]
    print(f"Dataset loaded with {len(dataset)} examples.")

    df = pd.DataFrame(dataset)

    # Get the list of all possible emotion labels
    emotion_labels = dataset.features['labels'].feature.names

    # Create one-hot encoded columns for each emotion
    # Initialize all emotion columns to 0
    for label in emotion_labels:
        df[f'emotion_{label}'] = 0

    # Fill in 1 for emotions present in each row
    for i, row in df.iterrows():
        if 'labels' in row and row['labels'] is not None:
            for idx in row['labels']:
                # Ensure the index is within bounds
                if 0 <= idx < len(emotion_labels):
                    df.at[i, f'emotion_{emotion_labels[idx]}'] = 1
        else:
            print(f"Warning: Row {i} has no 'labels' key or it's None. Skipping emotion encoding for this row.")


    # Select only the 'text' column and the new 'emotion_X' columns
    # Also add a placeholder 'is_fake' column, which will be 0 for this dataset
    selected_columns = ['text'] + [f'emotion_{l}' for l in emotion_labels]
    df['is_fake'] = 0 # GoEmotions are generally real conversations/posts

    # Filter out rows where 'text' might be missing or empty if any
    df = df.dropna(subset=['text']).copy()
    df = df[df['text'].str.strip() != ''].copy()

    print(f"Processed GoEmotions DataFrame shape: {df.shape}")
    return df[selected_columns + ['is_fake']]

if __name__ == "__main__":
    # This part runs only when data_loader.py is executed directly
    # Download and save the train split to data/raw/
    go_emotions_train_df = download_goemotions(split='train')
    go_emotions_train_df.to_csv("data/raw/goemotions_train.csv", index=False)
    print("GoEmotions train dataset saved to data/raw/goemotions_train.csv")

    # Download and save the validation split
    go_emotions_val_df = download_goemotions(split='validation')
    go_emotions_val_df.to_csv("data/raw/goemotions_val.csv", index=False)
    print("GoEmotions validation dataset saved to data/raw/goemotions_val.csv")

    # Download and save the test split
    go_emotions_test_df = download_goemotions(split='test')
    go_emotions_test_df.to_csv("data/raw/goemotions_test.csv", index=False)
    print("GoEmotions test dataset saved to data/raw/goemotions_test.csv")