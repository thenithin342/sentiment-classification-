import tensorflow as tf
from tensorflow.keras import layers, models


def get_model(vocab_size, embedding_dim=128, max_len=300, units=128, dropout_rate=0.5, model_type='lstm'):
    """
    Sentiment classification model supporting LSTM, GRU, and Bidirectional LSTM.
    """

    inputs = layers.Input(shape=(max_len,), name="input_tokens")

    # Embedding with masking (CRITICAL)
    x = layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        mask_zero=True,
        name="embedding"
    )(inputs)


    # Sequence processing layer
    if model_type == 'lstm':
        x = layers.LSTM(units, name="lstm")(x)
    elif model_type == 'gru':
        x = layers.GRU(units, name="gru")(x)
    elif model_type == 'bidirectional':
        x = layers.Bidirectional(layers.LSTM(units), name="bilstm")(x)
    else:
        raise ValueError(f"Invalid model_type: {model_type}")
    
    # Add Dropout to prevent overfitting
    x = layers.Dropout(dropout_rate, name="dropout")(x)

    # Classification head
    x = layers.Dense(64, activation="relu", name="dense")(x)

    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model
