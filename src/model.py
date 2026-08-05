"""1D-CNN + Transformer hybrid model for classifying hand landmarks into ASL signs."""
import tensorflow as tf
from tensorflow.keras import layers, models


def transformer_block(inputs, num_heads, ff_dim, dropout=0.1):
    attn_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=inputs.shape[-1])(inputs, inputs)
    attn_output = layers.Dropout(dropout)(attn_output)
    attn_output = layers.LayerNormalization(epsilon=1e-6)(attn_output + inputs)

    ffn_output = layers.Dense(ff_dim, activation="relu")(attn_output)
    ffn_output = layers.Dense(inputs.shape[-1])(ffn_output)
    ffn_output = layers.Dropout(dropout)(ffn_output)
    ffn_output = layers.LayerNormalization(epsilon=1e-6)(ffn_output + attn_output)

    return ffn_output


def build_cnn_transformer(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv1D(64, kernel_size=3, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Conv1D(128, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    seq_len = x.shape[1]
    embed_dim = x.shape[2]
    positions = tf.range(start=0, limit=seq_len, delta=1)
    positional_encoding = layers.Embedding(input_dim=seq_len, output_dim=embed_dim)(positions)
    x += positional_encoding

    x = transformer_block(x, num_heads=4, ff_dim=128, dropout=0.1)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return models.Model(inputs, outputs)
