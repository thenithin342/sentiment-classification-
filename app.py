import streamlit as st
import tensorflow as tf
import os
import time
from utils import load_vocab, preprocess_text, text_to_indices

import concurrent.futures

# Page Config
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="🎬",
    layout="wide"
)

# Constants
DATA_DIR = 'data'
MODEL_PATHS = {
    'LSTM': 'lstm_model.keras',
    'GRU': 'gru_model.keras',
    'Bi-LSTM': 'bilstm_model.keras'
}
MAX_LEN = 300
THRESHOLD = 0.5

@st.cache_resource
def load_resources():
    """
    Load models and vocabulary only once.
    """
    vocab_path = os.path.join(DATA_DIR, 'vocab.json')
    
    if not os.path.exists(vocab_path):
        st.error(f"Vocabulary not found at {vocab_path}. Please run training first.")
        return None, None
        
    models = {}
    for name, path in MODEL_PATHS.items():
        if os.path.exists(path):
            try:
                models[name] = tf.keras.models.load_model(path)
            except Exception as e:
                st.error(f"Failed to load {name} model from {path}: {e}")
        else:
            st.warning(f"Model {name} not found at {path}. run training.")

    if not models:
        return None, None

    word2idx = load_vocab(vocab_path)
    return models, word2idx

def predict(text, model, word2idx):
    if not text.strip():
        return None
        
    processed_text = preprocess_text(text)
    if not processed_text:
        return None
        
    padded_indices = text_to_indices(processed_text, word2idx, MAX_LEN)
    input_data = tf.expand_dims(padded_indices, 0)
    
    # training=False ensures deterministic inference (no dropout)
    prob = model(input_data, training=False).numpy()[0][0]
    return prob

def predict_all_models(text, models, word2idx):
    results = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_model = {executor.submit(predict, text, model, word2idx): name for name, model in models.items()}
        for future in concurrent.futures.as_completed(future_to_model):
            name = future_to_model[future]
            try:
                prob = future.result()
                results[name] = prob
            except Exception as e:
                results[name] = None
                print(f"Error predicting with {name}: {e}")
    return results

def analyze_importance(text, model, word2idx, original_score):
    """
    Perturbation Analysis: Remove one word at a time and measure impact.
    """
    words = text.split()
    if len(words) < 2:
        return []

    importances = []
    for i in range(len(words)):
        # Create a variant with one word missing
        truncated_text = " ".join(words[:i] + words[i+1:])
        new_score = predict(truncated_text, model, word2idx)
        
        # Impact = how much the score DROPPED or CHANGED
        # If original was high (0.9) and removing 'good' makes it 0.6, impact is 0.3 (Positive word)
        # If original was low (0.1) and removing 'bad' makes it 0.4, impact is -0.3 (Negative word)
        
        impact = original_score - new_score
        importances.append((words[i], impact))
        
    return importances

# UI Layout
st.title("🎬 Movie Review Sentiment Analyzer")
st.markdown("Enter a movie review below to see if it's **Positive** or **Negative**.")

# Sidebar for options
with st.sidebar:
    st.header("Settings")
    show_interpretability = st.checkbox("Show Explainability", value=True)
    st.info("Check 'Show Explainability' to see which words influenced the prediction.")

# Load resources
models, word2idx = load_resources()

if models and word2idx:
    # Input Area
    text_input = st.text_area("Your Review:", height=150, placeholder="E.g. The visuals were stunning, but the plot was weak...")

    if st.button("Analyze Sentiment", type="primary"):
        if text_input:
            with st.spinner("Analyzing with all models..."):
                results = predict_all_models(text_input, models, word2idx)

            # Display Results
            st.subheader("Model Predictions")
            cols = st.columns(len(results))
            
            # Determine overall sentiment for interpretability (using the Model with highest confidence or just the first one? Let's use LSTM or the first key)
            primary_model_name = 'LSTM' if 'LSTM' in models else list(models.keys())[0]
            primary_model_prob = results.get(primary_model_name)

            for idx, (name, prob) in enumerate(results.items()):
                with cols[idx]:
                    st.markdown(f"### {name}")
                    if prob is not None:
                        confidence = abs(prob - 0.5) * 2
                        if prob >= THRESHOLD:
                            st.success(f"POSITIVE")
                        else:
                            st.error(f"NEGATIVE")
                        
                        st.progress(float(prob))
                        st.caption(f"Score: {prob:.4f}")
                        st.caption(f"Confidence: {confidence:.2f}")
                    else:
                        st.warning("Failed")

            # Interpretability (using primary model)
            if show_interpretability and primary_model_prob is not None:
                st.markdown("---")
                st.subheader(f"🧐 Explainability (based on {primary_model_name})")
                
                with st.spinner(f"Calculating importance using {primary_model_name}..."):
                    importances = analyze_importance(text_input, models[primary_model_name], word2idx, primary_model_prob)
                
                if importances:
                    # Sort by absolute impact
                    importances.sort(key=lambda x: abs(x[1]), reverse=True)
                    top_n = importances[:5]
                    
                    st.write("### Top Influential Words")
                    for word, impact in top_n:
                        if impact > 0.05:
                            st.markdown(f"- **{word}**: pushed positive 🟩 (+{impact:.3f})")
                        elif impact < -0.05:
                            st.markdown(f"- **{word}**: pushed negative 🟥 ({impact:.3f})")
                        else:
                             st.markdown(f"- **{word}**: neutral ⬜ ({impact:.3f})")
                else:
                    st.info("Not enough words to analyze.")

        else:
            st.warning("Please enter some text first.")

else:
    st.info("Waiting for model resources available...")

