import pandas as pd
import os
from collections import Counter

def load_data(data_dir):
    """
    Loads train and test data from the specified directory.
    Expects train.csv and test.csv in the directory.
    """
    train_path = os.path.join(data_dir, 'train.csv')
    test_path = os.path.join(data_dir, 'test.csv')

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train file not found at {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test file not found at {test_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    return train_df, test_df


import re
import json

def preprocess_text(text):
    """
    Robust text preprocessing: lowercasing, removing HTML tags, and keeping only alphanumeric.
    """
    text = text.lower()
    text = re.sub(r"<br\s*/?>", " ", text) # Remove HTML line breaks
    text = re.sub(r"[^a-z0-9\s]", "", text) # Keep only alphanumeric and whitespace
    return text.strip()

def save_vocab(word2idx, path):
    """
    Saves the vocabulary to a JSON file.
    """
    with open(path, 'w') as f:
        json.dump(word2idx, f)

def load_vocab(path):
    """
    Loads the vocabulary from a JSON file.
    """
    with open(path, 'r') as f:
        return json.load(f)

def build_vocab(texts, max_vocab_size=20000):
    """
    Builds a vocabulary from a list of strings.
    """
    counter = Counter()
    for text in texts:
        counter.update(preprocess_text(text).split()) # Ensure we use the same preprocessing

    most_common = counter.most_common(max_vocab_size - 2)

    word2idx = {
        "<PAD>": 0,
        "<UNK>": 1
    }

    for i, (word, _) in enumerate(most_common, start=2):
        word2idx[word] = i


    return word2idx

def text_to_indices(text, word2idx, max_len):
    """
    Converts a text string to a list of indices and pads it.
    """
    tokens = text.split()
    indices = [word2idx.get(token, word2idx["<UNK>"]) for token in tokens]
    return pad_sequence(indices, max_len)


def pad_sequence(seq, max_len):
    """
    Pads the sequence to the maximum length.
    """
    if len(seq) >= max_len:
        return seq[:max_len]
    return seq + [0] * (max_len - len(seq))

import tensorflow as tf
import numpy as np


def get_vocab_size(word2idx):
    """
    Returns the size of the vocabulary.
    """
    return len(word2idx)

def get_tf_dataset(df, word2idx, max_len, batch_size=32, shuffle=True):

    """
    Converts a pandas DataFrame to a TensorFlow Dataset.
    Using vectorized access and re-using text_to_indices.
    """
    assert "text" in df.columns, "DataFrame must contain 'text' column"
    assert "label" in df.columns, "DataFrame must contain 'label' column"

    # IMDB reviews are long; 300 tokens capture most sentiment
    # We iterate over values to avoid slow iterrows()
    texts = df['text'].values
    labels = df['label'].values

    sequences = []
    
    for text in texts:
        # Re-use the existing logic!
        # Preprocessing is inside the loop but could be vectorized if we wanted to be super optimal
        # For now, this is much faster than iterrows
        processed_text = preprocess_text(text)
        padded = text_to_indices(processed_text, word2idx, max_len)
        sequences.append(padded)
    
    sequences = np.array(sequences, dtype=np.int32)
    # Ensure labels are float32 for BinaryCrossentropy(from_logits=True) or similar
    labels = np.array(labels, dtype=np.float32)
    
    dataset = tf.data.Dataset.from_tensor_slices((sequences, labels))
    
    if shuffle:
        # Cap shuffle buffer to avoid memory spikes
        buffer_size = min(len(df), 10000)
        dataset = dataset.shuffle(buffer_size=buffer_size)
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset



