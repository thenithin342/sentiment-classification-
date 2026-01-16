import argparse
import os
import tensorflow as tf
from utils import load_data, build_vocab, get_tf_dataset
from model import get_model

def main():
    parser = argparse.ArgumentParser(description='Train Sentiment Model (TensorFlow)')
    parser.add_argument('--data_dir', type=str, default='data', help='Path to data directory')
    parser.add_argument('--model_path', type=str, default='sentiment_model.keras', help='Path to save trained model')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--vocab_size', type=int, default=10000, help='Max vocabulary size')
    parser.add_argument('--max_len', type=int, default=300, help='Max sequence length')
    parser.add_argument('--model_type', type=str, default='lstm', choices=['lstm', 'gru', 'bidirectional'], help='Model type to train')
    parser.add_argument('--limit', type=int, default=None, help='Limit dataset size for testing')
    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.data_dir}...")
    train_df, test_df = load_data(args.data_dir)
    
    if args.limit:
        print(f"Limiting data to {args.limit} samples...")
        train_df = train_df.head(args.limit)
        test_df = test_df.head(args.limit)

    # Build vocab
    print("Building vocabulary...")
    # Concatenate train and test text to build full vocab or just use train?
    # Usually better to build on train only to avoid leakage, but for simple BoW/Embedding often whole corpus is used.

    # Let's stick to train only for strict correctness.
    word2idx = build_vocab(train_df['text'], max_vocab_size=args.vocab_size)
    print(f"Vocabulary size: {len(word2idx)}")
    
    # Save vocabulary
    from utils import save_vocab
    vocab_path = os.path.join(args.data_dir, 'vocab.json')
    save_vocab(word2idx, vocab_path)
    print(f"Vocabulary saved to {vocab_path}")

    # Create TF Datasets
    print("Creating TensorFlow datasets...")
    train_ds = get_tf_dataset(train_df, word2idx, args.max_len, args.batch_size, shuffle=True)
    val_ds = get_tf_dataset(test_df, word2idx, args.max_len, args.batch_size, shuffle=False)

    # Create model
    print("Creating model...")
    # Add 1 for the max index because indices are 0 to vocab_size+1 (actually len(word2idx) includes special tokens)
    # The Embedding layer needs input_dim = size of vocabulary + 1 (for 0 index/padding if not reserved or just max_index + 1)
    # Our word2idx values go up to len(word2idx)+1 potentially?
    # word2idx starts at 2. <PAD>=0, <UNK>=1.
    # Max index is roughly len(word2idx) - 1 + 2? No, len(word2idx) IS the size.
    # Actually, if word2idx has N items, the indices are 0..N-1?
    # Let's check build_vocab logic: <PAD>:0, <UNK>:1, then 2,3,4...
    # So max index used is len(word2idx)-1.

    # Max index is roughly len(word2idx) - 1 + 2? No, len(word2idx) IS the size.
    # Actually, if word2idx has N items, the indices are 0..N-1?
    # Let's check build_vocab logic: <PAD>:0, <UNK>:1, then 2,3,4...
    # So max index used is len(word2idx)-1.
    # Embedding input_dim should be len(word2idx)
    
    model = get_model(vocab_size=len(word2idx), max_len=args.max_len, model_type=args.model_type)
    
    model.compile(optimizer='adam',

                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    model.summary()

    # Train
    print("Training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=[tf.keras.callbacks.ModelCheckpoint(args.model_path, save_best_only=True)]
    )
    
    print(f"Model saved to {args.model_path}")

if __name__ == "__main__":
    main()
