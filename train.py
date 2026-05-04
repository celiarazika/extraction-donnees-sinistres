"""
Main training script for insurance claims extraction model.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Add src to path
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

from data_processor import process_pipeline
from model import create_model, train_model, evaluate_model

# Configuration
DATA_FILE = 'Insurance claims data.csv'
OUTPUT_DIR = 'data'
MODEL_DIR = 'models'

def main():
    """Execute full training pipeline."""
    
    print("="*60)
    print("PIPELINE DE TRAITEMENT ET D'ENTRAÎNEMENT")
    print("="*60)
    
    # 1. Preprocessing
    print("\n[1/3] Preprocessing des données...")
    processor, df_transformed = process_pipeline(DATA_FILE, OUTPUT_DIR)
    
    # 2. Prepare data for training
    print("\n[2/3] Préparation des données pour l'entraînement...")
    
    # Split into train/test
    X_train, X_test = train_test_split(df_transformed, test_size=0.2, random_state=42)
    
    # Split train into train/validation
    X_train, X_val = train_test_split(X_train, test_size=0.2, random_state=42)
    
    print(f"Train: {X_train.shape}")
    print(f"Validation: {X_val.shape}")
    print(f"Test: {X_test.shape}")
    
    # Save datasets
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    X_train.to_csv(os.path.join(OUTPUT_DIR, 'X_train.csv'), index=False)
    X_val.to_csv(os.path.join(OUTPUT_DIR, 'X_val.csv'), index=False)
    X_test.to_csv(os.path.join(OUTPUT_DIR, 'X_test.csv'), index=False)
    print(f"\nDatasets sauvegardés dans '{OUTPUT_DIR}'")
    
    # 3. Train model
    print("\n[3/3] Entraînement du modèle...")
    
    input_dim = X_train.shape[1]
    model = create_model(input_dim=input_dim)
    print(f"\nModèle créé avec {input_dim} features d'entrée")
    print(model.summary())
    
    # Train
    history = train_model(
        model, 
        X_train.values, 
        np.random.rand(len(X_train)),  # Placeholder labels - à adapter selon votre tâche
        X_val.values,
        np.random.rand(len(X_val)),
        epochs=50,
        batch_size=32,
        verbose=1
    )
    
    # Evaluate
    eval_results = evaluate_model(model, X_test.values, np.random.rand(len(X_test)))
    print(f"\nRésultats de test:")
    print(f"  Loss: {eval_results['loss']:.4f}")
    print(f"  Accuracy: {eval_results['accuracy']:.4f}")
    
    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(os.path.join(MODEL_DIR, 'insurance_model.h5'))
    print(f"\nModèle sauvegardé dans '{MODEL_DIR}'")
    
    print("\n" + "="*60)
    print("PIPELINE COMPLÉTÉ")
    print("="*60)

if __name__ == '__main__':
    main()
