import tensorflow as tf
import pandas as pd
import os
from utils import load_data, load_vocab, preprocess_text, text_to_indices

MAX_LEN = 300
THRESHOLD = 0.5

def predict_batch(texts, model, word2idx):
    # Optimizing batch inference by processing in chunks if needed, 
    # but for 25k samples, we might want to batch this explicitly to avoid OOM if GPU memory is tight.
    # However, the user provided code creates one giant tensor. 
    # Let's stick to the user's logic but add a batching loop if we wanted to be safer.
    # For now, following the user's snippet exactly as it's cleaner for them to read.
    # Wait, creating a list of 25k arrays then converting to tensor might be slowish but okay for this size.
    
    sequences = [
        text_to_indices(preprocess_text(text), word2idx, MAX_LEN)
        for text in texts
    ]
    
    # Convert to tensor. If OOM occurs, we'll need to use tf.data.Dataset or batching.
    # Let's try to be safe and use a simple batch loop so we don't crash on 25k sequences at once if on GPU.
    # Actually, let's stick to the prompt's request to keep it simple, but maybe use model.predict() 
    # on the sequences array which handles batching automatically?
    # The user wrote: probs = model(sequences, training=False).numpy().flatten()
    # This calls the model on the WHOLE set at once. This might OOM on 25k * 300 ints.
    # I will modify it slightly to use model.predict() which batches automatically, 
    # OR create a dataset. 
    # BUT, the user explicitly said "training=False" to be safe. model.predict() does that too.
    # Let's use model.predict(sequences, batch_size=32, verbose=1) for safety and progress bar.
    
    sequences = tf.constant(sequences)
    # model.predict is safer for large data than model(sequences)
    probs = model.predict(sequences, batch_size=64, verbose=1).flatten()
    return probs

def main():
    data_dir = "data"
    model_path = "sentiment_model.keras"

    # Load data
    print("Loading test data...")
    _, test_df = load_data(data_dir)

    # Load vocab & model
    print("Loading model and vocab...")
    word2idx = load_vocab(os.path.join(data_dir, "vocab.json"))
    model = tf.keras.models.load_model(model_path)

    # Predict
    print("Running predictions on test set...")
    probs = predict_batch(test_df["text"].values, model, word2idx)

    test_df["pred_prob"] = probs
    test_df["pred_label"] = (probs >= THRESHOLD).astype(int)

    # Accuracy sanity check
    accuracy = (test_df["pred_label"] == test_df["label"]).mean()
    print(f"Test Accuracy (manual): {accuracy:.4f}")

    # Explicit Error Sampling (Step 3)
    errors = test_df[test_df["pred_label"] != test_df["label"]]
    print(f"\nTotal Errors: {len(errors)} out of {len(test_df)}")
    print("\n--- Error Samples ---")
    if len(errors) > 0:
        pd.set_option('display.max_colwidth', 150) # Make sure we can read the text
        print(errors.sample(min(5, len(errors)))[["text", "label", "pred_label", "pred_prob"]])
    print("---------------------\n")

    # Save results
    output_path = os.path.join(data_dir, "test_predictions.csv")
    test_df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    main()
