

import argparse
import tensorflow as tf
import os
from utils import load_data, get_tf_dataset, load_vocab

def main():
    parser = argparse.ArgumentParser(description='Evaluate Sentiment Model (TensorFlow)')
    parser.add_argument('--data_dir', type=str, default='data', help='Path to data directory')
    parser.add_argument('--model_path', type=str, default='sentiment_model.keras', help='Path to load trained model')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--max_len', type=int, default=300, help='Max sequence length (must match training)')
    args = parser.parse_args()


    # Load data
    print(f"Loading data from {args.data_dir}...")
    _, test_df = load_data(args.data_dir)

    # Load vocab
    vocab_path = os.path.join(args.data_dir, 'vocab.json')
    print(f"Loading vocabulary from {vocab_path}...")
    word2idx = load_vocab(vocab_path)
    
    # Safety Check
    assert len(word2idx) > 0, "Loaded vocabulary is empty!"
    
    print(f"Vocabulary size: {len(word2idx)}")
    print(f"Max sequence length: {args.max_len}")



    # Create TF Dataset for test
    print("Creating TensorFlow test dataset...")
    test_ds = get_tf_dataset(test_df, word2idx, args.max_len, args.batch_size, shuffle=False)

    # Load model
    print(f"Loading model from {args.model_path}...")
    model = tf.keras.models.load_model(args.model_path)

    # Evaluate
    print("Evaluating...")
    loss, accuracy = model.evaluate(test_ds)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
