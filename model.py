import tensorflow as tf
from tensorflow.keras import layers, models


def get_model(vocab_size, embedding_dim=128, max_len=300, lstm_units=128, dropout_rate=0.5):
    """
    LSTM-based sentiment classification model
    """

    inputs = layers.Input(shape=(max_len,), name="input_tokens")

    # Embedding with masking (CRITICAL)
    x = layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        mask_zero=True,
        name="embedding"
    )(inputs)


    # LSTM processes sequence step-by-step
    x = layers.LSTM(
        lstm_units,
        name="lstm"
    )(x)
    
    # Add Dropout to prevent overfitting
    x = layers.Dropout(dropout_rate, name="dropout")(x)

    # Classification head
    x = layers.Dense(64, activation="relu", name="dense")(x)

    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model
