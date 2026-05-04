"""
Deep Learning model for insurance claims data extraction.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


def create_model(input_dim, output_dim=1, hidden_units=[256, 128, 64]):
    """
    Create a neural network model.
    
    Args:
        input_dim: Number of input features
        output_dim: Number of output units (1 for regression/binary classification)
        hidden_units: List of hidden layer sizes
    
    Returns:
        Compiled Keras model
    """
    model = keras.Sequential([
        layers.Dense(hidden_units[0], activation='relu', input_dim=input_dim),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
    ])
    
    for units in hidden_units[1:]:
        model.add(layers.Dense(units, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.3))
    
    model.add(layers.Dense(output_dim, activation='sigmoid' if output_dim == 1 else 'softmax'))
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy' if output_dim == 1 else 'categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def train_model(model, X_train, y_train, X_val=None, y_val=None, 
                epochs=50, batch_size=32, verbose=1):
    """
    Train the model.
    
    Args:
        model: Compiled Keras model
        X_train: Training features
        y_train: Training labels
        X_val: Validation features (optional)
        y_val: Validation labels (optional)
        epochs: Number of training epochs
        batch_size: Batch size for training
        verbose: Verbosity level
    
    Returns:
        Training history
    """
    validation_data = None
    if X_val is not None and y_val is not None:
        validation_data = (X_val, y_val)
    
    history = model.fit(
        X_train, y_train,
        validation_data=validation_data,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor='val_loss' if validation_data else 'loss',
                patience=5,
                restore_best_weights=True
            )
        ]
    )
    
    return history


def evaluate_model(model, X_test, y_test):
    """Evaluate model on test set."""
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    return {'loss': loss, 'accuracy': accuracy}


def predict(model, X):
    """Make predictions."""
    return model.predict(X, verbose=0)
