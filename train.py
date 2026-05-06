"""
LLM-based claim description generator.
Loads insurance claims data and generates descriptions using an LLM.
"""

import os
import sys
from pathlib import Path
import time  # Ajouter timing

import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Charger les variables d'environnement depuis .env
load_dotenv()

# Add src to path
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

from data_processor import DataProcessor
from model import create_generator

# Configuration
DATA_FILE = 'Insurance claims data.csv'
OUTPUT_DIR = 'data'
LLM_MODEL = 'ollama'  # Ollama only - local, free LLM

# Vérifier que Ollama est accessible
if LLM_MODEL == 'ollama':
    import requests
    try:
        requests.get("http://localhost:11434/api/tags", timeout=2)
        print("Ollama trouvé sur localhost:11434")
    except:
        raise ValueError(
            "Ollama n'est pas accessible!\n"
            "Étapes:\n"
            "  1. Téléchargez Ollama: https://ollama.ai\n"
            "  2. Lancez: ollama serve\n"
            "  3. Dans un autre terminal: ollama pull mistral\n"
            "  4. Puis relancez ce script\n"
        )

def main():
    """Execute the claim description generation pipeline."""
    
    print("="*60)
    print("PIPELINE DE GÉNÉRATION DE DESCRIPTIONS DE SINISTRES (LLM)")
    print("="*60)
    
    # 1. Load and preprocess data
    print("\n[1/3] Chargement et preprocessing des données...")
    processor = DataProcessor()
    df = processor.load_data(DATA_FILE)
    
    print(f"\n Données chargées: {df.shape}")
    print(f"Colonnes: {list(df.columns)[:5]}...")
    
    # Analyze quality
    processor.analyze_quality(df)
    
    # Clean data
    df_clean = processor.clean_data(df)
    
    # 2. Initialize LLM
    print(f"\n[2/3] Initialisation du LLM ({LLM_MODEL})...")
    try:
        generator = create_generator(model_name=LLM_MODEL)
        print(f"Générateur LLM prêt")
    except Exception as e:
        print(f"Erreur lors du chargement du LLM: {e}")
        print("Vérifiez que vous avez installé: pip install transformers torch")
        return
    
    # 3. Generate descriptions
    print(f"\n[3/3] Génération de descriptions...")
    
    # Convert dataframe to list of dictionaries
    claims_data = df_clean.head(10).to_dict('records')  # Start with first 10 for demo
    
    # Generate descriptions
    descriptions = []
    start_time = time.time()
    
    for i, claim in enumerate(claims_data):
        iter_start = time.time()
        print(f"\n📝 Sinistre {i+1}/{len(claims_data)}")
        
        try:
            description = generator.generate(claim, max_length=300)
            descriptions.append(description)
            iter_time = time.time() - iter_start
            print(f"⏱️  Temps: {iter_time:.1f}s")
            print(f"\n📄 Description générée:\n{description}")
        except Exception as e:
            print(f"❌ Erreur lors de la génération: {e}")
            descriptions.append("Erreur de génération")
    
    total_time = time.time() - start_time
    print(f"\n⏱️  Temps total: {total_time:.1f}s ({total_time/len(claims_data):.1f}s par sinistre)")
    
    # 4. Save results
    print("\n" + "="*60)
    print("RÉSULTATS")
    print("="*60)
    
    # Create output dataframe
    results_df = df_clean.head(len(descriptions)).copy()
    results_df['description'] = descriptions
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file = os.path.join(OUTPUT_DIR, 'claims_with_descriptions.csv')
    results_df.to_csv(output_file, index=False)
    
    print(f"\n Résultats sauvegardés dans: {output_file}")
    print(f"   Nombre de sinistres avec descriptions: {len(results_df)}")
    
    # Display sample
    print("\n Exemples:")
    for idx in range(min(3, len(results_df))):
        print(f"\nSinistre #{idx+1}")
        print(f"  Description: {results_df.iloc[idx]['description']}")
    
    print("\n" + "="*60)
    print("PIPELINE COMPLÉTÉ")
    print("="*60)
    print("\n Prochaine étape: streamlit run app.py")

if __name__ == '__main__':
    main()
