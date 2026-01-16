import tensorflow as tf
import os
import pandas as pd
from utils import load_data, load_vocab, get_tf_dataset
from model import get_model

# Constants
DATA_DIR = 'data'
VOCAB_PATH = os.path.join(DATA_DIR, 'vocab.json')
MAX_LEN = 300
EPOCHS = 2  # Keep low for tuning speed

def run_experiment(config, train_ds, val_ds, vocab_size):
    print(f"\nRunning experiment: {config}")
    
    model = get_model(
        vocab_size=vocab_size, 
        embedding_dim=config['embedding_dim'], 
        max_len=MAX_LEN,
        lstm_units=config['lstm_units'],
        dropout_rate=config['dropout']
    )
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        verbose=1
    )
    
    best_val_acc = max(history.history['val_accuracy'])
    return best_val_acc

def main():
    # Load Data
    train_df, test_df = load_data(DATA_DIR)
    word2idx = load_vocab(VOCAB_PATH)
    vocab_size = len(word2idx)
    
    # Create Datasets (Fixed Batch Size 32 for comparison)
    train_ds = get_tf_dataset(train_df, word2idx, MAX_LEN, batch_size=32, shuffle=True)
    val_ds = get_tf_dataset(test_df, word2idx, MAX_LEN, batch_size=32, shuffle=False)
    
    # Search Space
    configs = [
        {'embedding_dim': 64, 'lstm_units': 64, 'dropout': 0.5},
        {'embedding_dim': 128, 'lstm_units': 128, 'dropout': 0.5}, # Baseline
        {'embedding_dim': 128, 'lstm_units': 128, 'dropout': 0.3}, # Lower dropout
        {'embedding_dim': 256, 'lstm_units': 128, 'dropout': 0.5}, # Larger embedding
    ]
    
    results = []
    
    for config in configs:
        acc = run_experiment(config, train_ds, val_ds, vocab_size)
        res = config.copy()
        res['val_accuracy'] = acc
        results.append(res)
        print(f"Result: {res}")
        
    # Summary
    print("\n--- Tuning Results ---")
    results_df = pd.DataFrame(results)
    print(results_df.sort_values(by='val_accuracy', ascending=False))
    
    results_df.to_csv(os.path.join(DATA_DIR, 'tuning_results.csv'), index=False)

if __name__ == "__main__":
    main()
