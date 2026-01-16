import tensorflow as tf
import os
import argparse
from utils import load_vocab, preprocess_text, text_to_indices

THRESHOLD = 0.5

def predict(text, model, word2idx, max_len=300):
    # Preprocess
    processed_text = preprocess_text(text)
    
    if len(processed_text.strip()) == 0:
        raise ValueError("Input text is empty after preprocessing.")

    # Convert to indices
    padded_indices = text_to_indices(processed_text, word2idx, max_len)
    # Add batch dimension
    input_data = tf.expand_dims(padded_indices, 0)
    
    # Predict with training=False to ensure deterministic behavior (no dropout)
    prob = model(input_data, training=False).numpy()[0][0]
    return prob

def main():
    parser = argparse.ArgumentParser(description='Predict Sentiment')
    parser.add_argument('text', type=str, help='Text to classify')
    parser.add_argument('--data_dir', type=str, default='data', help='Path to data directory (for vocab)')
    parser.add_argument('--model_path', type=str, default='sentiment_model.keras', help='Path to model')
    parser.add_argument('--max_len', type=int, default=300, help='Max sequence length')
    args = parser.parse_args()

    # Load resources
    vocab_path = os.path.join(args.data_dir, 'vocab.json')
    if not os.path.exists(vocab_path):
        print(f"Error: Vocabulary not found at {vocab_path}. Run train.py first.")
        return
        
    print("Loading resources...")
    word2idx = load_vocab(vocab_path)
    model = tf.keras.models.load_model(args.model_path)
    
    try:
        # Predict
        probability = predict(args.text, model, word2idx, args.max_len)
        
        print("-" * 30)
        print(f"Input: {args.text}")
        print(f"Sentiment Probability (Positive): {probability:.4f}")
        
        if probability >= THRESHOLD:
            print("Prediction: POSITIVE")
        else:
            print("Prediction: NEGATIVE")
            
        # Interpretation
        confidence = abs(probability - 0.5) * 2
        print(f"Confidence: {confidence:.2f}")
        print("-" * 30)
        
    except ValueError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
