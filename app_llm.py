"""
Streamlit web interface for insurance claims LLM description generator.
Run with: streamlit run app_llm.py
"""

import os
import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np

# Add src to path
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

from data_processor import DataProcessor
from model import create_generator

# Configuration
LLM_MODEL = 'gpt2'

@st.cache_resource
def load_generator():
    """Load the LLM generator (cached for performance)."""
    try:
        return create_generator(model_name=LLM_MODEL)
    except Exception as e:
        st.error(f"Erreur lors du chargement du LLM: {e}")
        return None

def main():
    st.set_page_config(page_title="Insurance Claims AI", layout="wide")
    
    st.title("🚨 Générateur de Descriptions de Sinistres avec IA")
    st.markdown("Transformez vos données brutes de sinistres en descriptions cohérentes grâce à un LLM")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    page = st.sidebar.radio(
        "Sélectionnez une section",
        ["🏠 Accueil", "✍️ Générer Description", "📁 Batch Processing", "📊 Informations"]
    )
    
    # Load generator
    generator = load_generator()
    if generator is None:
        st.error("❌ Impossible de charger le modèle LLM")
        return
    
    # Home page
    if page == "🏠 Accueil":
        st.header("Bienvenue 👋")
        st.write("""
        Cette application utilise un **Language Model (LLM)** pour générer des descriptions
        cohérentes et naturelles à partir des données de sinistres d'assurance.
        
        **Fonctionnalités:**
        - 📝 Génération de descriptions pour sinistres individuels
        - 📁 Traitement batch de fichiers CSV
        - 🎯 Utilise GPT-2 (ou un LLM configurable)
        - ⚡ Génération rapide et locale
        
        **Cas d'usage:**
        - 📋 Automatiser la documentation des sinistres
        - 🔍 Générer des résumés pour les analystes
        - 📊 Enrichir les données avec du texte descriptif
        """)
        
        st.info("ℹ️ Utilisez le menu latéral pour naviguer vers les autres sections")
    
    # Generation page
    elif page == "✍️ Générer Description":
        st.header("Génération de Description")
        
        st.write("Entrez les caractéristiques d'un sinistre pour générer une description:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Données du Sinistre")
            claim_data = {}
            
            claim_data["Type"] = st.selectbox(
                "Type de sinistre",
                ["Accident Auto", "Sinistre Habitation", "Vol", "Dégâts des Eaux", "Autre"]
            )
            
            claim_data["Montant"] = st.number_input("Montant (€)", min_value=0, value=5000)
            
            claim_data["Statut"] = st.selectbox(
                "Statut",
                ["Ouvert", "En cours", "Fermé", "En litige"]
            )
            
            claim_data["Cause"] = st.text_input("Cause du sinistre", placeholder="Ex: collision")
            
            claim_data["Description_brute"] = st.text_area(
                "Description brute",
                placeholder="Décrivez le sinistre en quelques mots...",
                height=100
            )
        
        with col2:
            st.subheader("🤖 Description Générée")
            
            if st.button("🚀 Générer Description", key="gen_single"):
                with st.spinner("Génération en cours..."):
                    try:
                        description = generator.generate(claim_data, max_length=120)
                        st.success("✅ Description générée!")
                        st.write(description)
                        
                        # Copy button
                        st.code(description, language="markdown")
                    except Exception as e:
                        st.error(f"❌ Erreur: {e}")
    
    # Batch processing page
    elif page == "📁 Batch Processing":
        st.header("Traitement par Batch")
        
        st.write("Chargez un fichier CSV pour générer des descriptions pour tous les sinistres")
        
        uploaded_file = st.file_uploader("Choisissez un fichier CSV", type=['csv'])
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            
            st.write("**Données chargées:**")
            st.dataframe(df.head())
            
            # Select columns to use
            cols_to_use = st.multiselect(
                "Sélectionnez les colonnes à utiliser pour la génération",
                df.columns.tolist(),
                default=df.columns.tolist()[:5]  # Default first 5 columns
            )
            
            num_rows = st.slider(
                "Nombre de sinistres à traiter",
                min_value=1,
                max_value=len(df),
                value=min(10, len(df))
            )
            
            if st.button("🔄 Traiter et Générer Descriptions"):
                with st.spinner(f"Génération de {num_rows} descriptions..."):
                    try:
                        df_subset = df[cols_to_use].head(num_rows)
                        
                        descriptions = []
                        progress_bar = st.progress(0)
                        
                        for idx, row in df_subset.iterrows():
                            # Create claim data dict
                            claim_data = row.to_dict()
                            
                            # Generate description
                            description = generator.generate(claim_data, max_length=100)
                            descriptions.append(description)
                            
                            # Update progress
                            progress = (idx + 1) / num_rows
                            progress_bar.progress(progress)
                        
                        # Add descriptions to dataframe
                        df_results = df.head(num_rows).copy()
                        df_results['description_generee'] = descriptions
                        
                        st.success(f"✅ {len(descriptions)} descriptions générées!")
                        st.dataframe(df_results[['description_generee']])
                        
                        # Download option
                        csv = df_results.to_csv(index=False)
                        st.download_button(
                            label="📥 Télécharger résultats",
                            data=csv,
                            file_name="claims_with_descriptions.csv",
                            mime="text/csv"
                        )
                    except Exception as e:
                        st.error(f"❌ Erreur lors du traitement: {e}")
    
    # Info page
    elif page == "📊 Informations":
        st.header("Informations du Système")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Modèle LLM", LLM_MODEL)
        
        with col2:
            st.metric("Type", "Génération de Texte")
        
        with col3:
            st.metric("Statut", "✅ Prêt")
        
        st.write("**Architecture:**")
        st.write("""
        - **Frontend:** Streamlit (interface web)
        - **Backend:** HuggingFace Transformers
        - **Modèle:** GPT-2 (ou configurable)
        - **Langage:** Python avec TensorFlow/PyTorch
        """)
        
        st.write("**Flux de traitement:**")
        st.write("""
        1. Utilisateur entre données brutes du sinistre
        2. Application crée un prompt pour le LLM
        3. LLM génère une description naturelle
        4. Résultat affiché/téléchargé
        """)

if __name__ == '__main__':
    main()
