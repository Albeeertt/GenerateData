from typing import List

import tensorflow as tf
from sklearn.model_selection import train_test_split
from pandas import DataFrame
# from transformers import TFLongformerModel, LongformerConfig

# En 1D no necesito identificadores o sí.
def cnn1d_improvement(num_classes, kernel_size, dropout : float = 0.2):
    model = tf.keras.Sequential([
        tf.layers.Input(shape=(num_classes,)),

        tf.keras.layers.Conv1D(
            filters = 16,
            kernel_size = kernel_size,
            activation = None,
            padding = 'same'
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.SpatialDropout1D(dropout),

        tf.keras.layers.Conv1D(
            filters = 32,
            kernel_size = kernel_size,
            activation = None,
            padding = "same"
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.SpatialDropout1D(dropout),

        tf.keras.layers.Conv1D(
            filters = 64,
            kernel_size = kernel_size,
            activation = None,
            padding = "same"
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.SpatialDropout1D(dropout),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(2688, activation='relu'),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(164, activation='relu'),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.Dense(num_classes, activation='softmax')

    ])

    return model

def cnn2d_improvement(num_kmers, num_classes, kernel_size, dropout : float = 0.2):
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(num_kmers, num_classes, 1)),

        tf.keras.layers.Conv2D(
            filters = 16,
            kernel_size = kernel_size,
            activation = None,
            padding = 'same'
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.SpatialDropout2D(dropout),

        tf.keras.layers.Conv2D(
            filters = 32,
            kernel_size = kernel_size,
            activation = None,
            padding = "same"
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.SpatialDropout2D(dropout),

        tf.keras.layers.Conv2D(
            filters = 64,
            kernel_size = kernel_size,
            activation = None,
            padding = "same"
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.SpatialDropout2D(dropout),

        tf.keras.layers.Conv2D(
            filters = 128,
            kernel_size = kernel_size,
            activation = None,
            padding = "same"
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.SpatialDropout2D(dropout),

        tf.keras.layers.Conv2D(
            filters = 256,
            kernel_size = kernel_size,
            activation = None,
            padding = "same"
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.SpatialDropout2D(dropout),

        tf.keras.layers.Conv2D(filters=1, kernel_size=(3, 3), padding="same"), 
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.SpatialDropout2D(dropout),
        tf.keras.layers.Reshape((num_kmers, num_classes)),

        tf.keras.layers.Softmax(axis=-1)

    ])

    return model

# Hazlo gandul
def autoencoder():
    return None;

def hibrid_model_cnnTransformer(num_kmers, num_classes, kernel_size, num_heads: int, expand_ffn: int = 2, dropout : float = 0.2):

    inputs = tf.keras.Input(shape=(num_kmers, num_classes, 1))

    conv = tf.keras.layers.Conv2D(
        filters = 16,
        kernel_size = kernel_size,
        activation = None,
        padding = 'same'
    )(inputs)
    
    batch = tf.keras.layers.BatchNormalization()(conv)
    relu = tf.keras.layers.ReLU()(batch)
    drop = tf.keras.layers.SpatialDropout2D(dropout)(relu)

    conv = tf.keras.layers.Conv2D(
        filters = 32,
        kernel_size = kernel_size,
        strides = 2,
        activation = None,
        padding = "same"
    )(drop)
    batch = tf.keras.layers.BatchNormalization()(conv)
    relu = tf.keras.layers.ReLU()(batch)
    drop = tf.keras.layers.SpatialDropout2D(dropout)(relu)

    conv = tf.keras.layers.Conv2D(
        filters = 64,
        kernel_size = kernel_size,
        activation = None,
        padding = "same"
    )(drop)
    batch = tf.keras.layers.BatchNormalization()(conv)
    relu = tf.keras.layers.ReLU()(batch)
    drop = tf.keras.layers.SpatialDropout2D(dropout)(relu)

    conv = tf.keras.layers.Conv2D(
        filters = 128,
        kernel_size = kernel_size,
        strides = 2,
        activation = None,
        padding = "same"
    )(drop)
    batch = tf.keras.layers.BatchNormalization()(conv)
    relu = tf.keras.layers.ReLU()(batch)
    drop = tf.keras.layers.SpatialDropout2D(dropout)(relu)

    x = tf.keras.layers.Reshape((-1, drop.shape[-1]))(drop)

    attn_output = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=128, dropout=dropout)(x, x)
    ffn = tf.keras.layers.Dense(expand_ffn*128, activation='relu')(attn_output)
    ffn = tf.keras.layers.Dropout(dropout)(ffn)
    ffn = tf.keras.layers.Dense(128, activation='relu')(ffn)
    ffn = tf.keras.layers.LayerNormalization()(ffn)

    attn_output = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=128, dropout=dropout)(ffn, ffn)
    ffn = tf.keras.layers.Dense(expand_ffn*128, activation='relu')(attn_output)
    ffn = tf.keras.layers.Dropout(dropout)(ffn)
    ffn = tf.keras.layers.Dense(128, activation='relu')(ffn)
    ffn = tf.keras.layers.LayerNormalization()(ffn)

    attn_output = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=128, dropout=dropout)(ffn, ffn)
    ffn = tf.keras.layers.Dense(expand_ffn*128, activation='relu')(attn_output)
    ffn = tf.keras.layers.Dropout(dropout)(ffn)
    ffn = tf.keras.layers.Dense(128, activation='relu')(ffn)
    ffn = tf.keras.layers.LayerNormalization()(ffn)

    # attn_output = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=128, dropout=dropout)(ffn, ffn)
    # ffn = tf.keras.layers.Dense(expand_ffn*128, activation='relu')(attn_output)
    # ffn = tf.keras.layers.Dropout(dropout)(ffn)
    # ffn = tf.keras.layers.Dense(128, activation='relu')(ffn)
    # ffn = tf.keras.layers.LayerNormalization()(ffn)

    # -------
    
    ffn = ffn[:, :, tf.newaxis, :] 

    # x = tf.keras.layers.Conv2DTranspose(64, kernel_size, strides=2, padding="same")(ffn)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.ReLU()(x)
    # x = tf.keras.layers.SpatialDropout2D(dropout)(x)

    x = tf.keras.layers.Conv2DTranspose(32, kernel_size, strides=2, padding="same")(ffn)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.SpatialDropout2D(dropout)(x)

    # x = tf.keras.layers.Conv2DTranspose(16, kernel_size, strides=2, padding="same")(ffn)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.ReLU()(x)
    # x = tf.keras.layers.SpatialDropout2D(dropout)(x)

    x = tf.keras.layers.Conv2DTranspose(1, kernel_size, strides=2, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.SpatialDropout2D(dropout)(x)

    # -------

    x = tf.keras.layers.Cropping2D(((0, 0), (0, 1)))(x)  # ahora shape: (batch, 16384, 3, 1)

    # Detalle muy importante, es necesario usar layers porque sino no son compatibles con keras :)
    x = tf.keras.layers.Lambda(lambda t: tf.squeeze(t, axis=-1))(x)

    output = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=output)
    return model


def transformer_complete_table(num_classes: int, num_heads: int, key_dim: int, expand_ffn: int = 2, dropout: float = 0.2 ):

    inputs = tf.keras.Input(shape=(None,num_classes))

    attn_output = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=dropout)(inputs, inputs)
    ffn = tf.keras.layers.Dense(expand_ffn*key_dim, activation='relu')(attn_output)
    ffn = tf.keras.layers.Dropout(dropout)(ffn)
    ffn = tf.keras.layers.Dense(key_dim, activation='relu')(ffn)
    ffn = tf.keras.layers.LayerNormalization()(ffn)


    attn_output = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=dropout)(ffn, ffn)
    ffn = tf.keras.layers.Dense(expand_ffn*key_dim, activation='relu')(attn_output)
    ffn = tf.keras.layers.Dropout(dropout)(ffn)
    ffn = tf.keras.layers.Dense(key_dim, activation='relu')(ffn)
    ffn = tf.keras.layers.LayerNormalization()(ffn)

    output = tf.keras.layers.Dense(num_classes, activation="softmax")(ffn)

    model = tf.keras.models.Model(inputs=inputs, outputs=output)
    return model

def transformer(num_classes: int, num_total_kmers: int, embedding_dim: int, num_features: int, num_heads: int, key_dim: int, expand_ffn: int = 2, dropout: float = 0.2 ):
    
    input_id = tf.keras.Input(shape=(None,))
    embedding_by_id = tf.keras.layers.Embedding(input_dim = num_total_kmers, output_dim = embedding_dim, embeddings_initializer='uniform')(input_id)

    input_freq = tf.keras.Input(shape=(None, num_features))

    combined = tf.keras.layers.Concatenate()([embedding_by_id, input_freq])


    attn_output = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=dropout)(combined, combined) 
    ffn = tf.keras.layers.Dense(expand_ffn*key_dim, activation='relu')(attn_output)
    ffn = tf.keras.layers.Dropout(dropout)(ffn)
    ffn = tf.keras.layers.Dense(key_dim, activation='relu')(ffn)
    ffn = tf.keras.layers.LayerNormalization()(ffn)


    attn_output = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=dropout)(ffn, ffn) 
    ffn = tf.keras.layers.Dense(expand_ffn*key_dim, activation='relu')(attn_output)
    ffn = tf.keras.layers.Dropout(dropout)(ffn)
    ffn = tf.keras.layers.Dense(key_dim, activation='relu')(ffn)
    ffn = tf.keras.layers.LayerNormalization()(ffn)

    output = tf.keras.layers.Dense(num_classes, activation="softmax")(ffn)

    model = tf.keras.models.Model(inputs=[input_id, input_freq], outputs=output)
    return model

def model_compile(model, optimizer, loss, metrics):
    model.compile(
        optimizer = optimizer,
        loss = loss,
        metrics = metrics
    )
