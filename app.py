import streamlit as st
import tensorflow as tf
import os
import time
from utils import load_vocab, preprocess_text, text_to_indices

# Page Config
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)

# Constants
DATA_DIR = 'data'
MODEL_PATH = 'sentiment_model.keras'
MAX_LEN = 300
THRESHOLD = 0.5

@st.cache_resource
def load_resources():
    """
    Load model and vocabulary only once.
    """
    vocab_path = os.path.join(DATA_DIR, 'vocab.json')
    
    if not os.path.exists(vocab_path):
        st.error(f"Vocabulary not found at {vocab_path}. Please run training first.")
        return None, None
        
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found at {MODEL_PATH}. Please run training first.")
        return None, None

    word2idx = load_vocab(vocab_path)
    model = tf.keras.models.load_model(MODEL_PATH)
    return model, word2idx

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
model, word2idx = load_resources()

if model and word2idx:
    # Input Area
    text_input = st.text_area("Your Review:", height=150, placeholder="E.g. The visuals were stunning, but the plot was weak...")

    if st.button("Analyze Sentiment", type="primary"):
        if text_input:
            with st.spinner("Analyzing..."):
                # Simulate a tiny delay for effect (optional, implies 'thinking')
                # time.sleep(0.3) 
                probability = predict(text_input, model, word2idx)

            if probability is not None:
                confidence = abs(probability - 0.5) * 2
                
                # Dynamic Color based on sentiment
                if probability >= THRESHOLD:
                    st.success(f"### Prediction: POSITIVE (Confidence: {confidence:.2f})")
                    if probability > 0.8:
                        st.balloons()
                else:
                    st.error(f"### Prediction: NEGATIVE (Confidence: {confidence:.2f})")
                
                # Gauge / Progress Bar
                st.progress(float(probability))
                st.caption(f"Score: {probability:.4f} (0 = Negative, 1 = Positive)")
                
                # Metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Sentiment Score", f"{probability:.1%}")
                with col2:
                    st.metric("Certainty", f"{confidence:.1%}")

                # Interpretability
                if show_interpretability:
                    with st.spinner("Calculating word importance..."):
                        importances = analyze_importance(text_input, model, word2idx, probability)
                    
                    st.markdown("---")
                    st.subheader("🧐 Why this prediction?")
                    st.markdown("This chart shows how much each word contributed to the **original** score.")
                    
                    # Sort by absolute impact
                    importances.sort(key=lambda x: abs(x[1]), reverse=True)
                    top_n = importances[:5] # Top 5
                    
                    # Prepare for chart
                    imp_data = {"Word": [x[0] for x in top_n], "Impact": [x[1] for x in top_n]}
                    
                    # Simple bar chart using st.bar_chart? 
                    # Prefer st.write with colored text for better intuition
                    
                    st.write("### Top Influential Words")
                    for word, impact in top_n:
                        # If impact is POSITIVE, it means removing it LOWERED the score -> The word was pulling it UP (Positive)
                        # If impact is NEGATIVE, it means removing it RAISED the score -> The word was pulling it DOWN (Negative)
                        
                        if impact > 0.05:
                            st.markdown(f"- **{word}**: pushed score **UP** (Positive signal) 🟩 (+{impact:.3f})")
                        elif impact < -0.05:
                            st.markdown(f"- **{word}**: pushed score **DOWN** (Negative signal) 🟥 ({impact:.3f})")
                        else:
                             st.markdown(f"- **{word}**: neutral influence ⬜ ({impact:.3f})")

            else:
                st.warning("Input text was empty or valid characters were removed during preprocessing.")
        else:
            st.warning("Please enter some text first.")

else:
    st.info("Waiting for model resources available...")

