"""
Streamlit web interface for insurance claims model.
Run with: streamlit run app.py
"""

import os
import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
from tensorflow import keras

# Add src to path
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

from data_processor import DataProcessor

# Configuration
MODEL_PATH = 'models/insurance_model.h5'
MODELS_DIR = 'models'

@st.cache_resource
def load_model():
    """Load the trained model."""
    if os.path.exists(MODEL_PATH):
        return keras.models.load_model(MODEL_PATH)
    return None

@st.cache_resource
def load_processor():
    """Load the data processor with fitted preprocessors."""
    processor = DataProcessor()
    processor.load_preprocessors(MODELS_DIR)
    return processor

def main():
    st.set_page_config(page_title="Insurance Claims AI", layout="wide")
    
    st.title("🏥 Extraction de Données de Sinistres d'Assurance")
    st.markdown("---")
    
    # Load model and processor
    model = load_model()
    processor = load_processor()
    
    if model is None:
        st.error("❌ Modèle non trouvé. Veuillez d'abord exécuter `python train.py`")
        return
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Sélectionnez une section",
        ["Accueil", "Inférence", "Upload & Prédiction", "Statistiques"]
    )
    
    # Home page
    if page == "Accueil":
        st.header("Bienvenue")
        st.write("""
        Cette application utilise un modèle de Deep Learning pour extraire et analyser
        les données de sinistres à partir de contrats d'assurance.
        
        **Fonctionnalités:**
        - 🔍 Analyse des données brutes
        - 🤖 Prédictions basées sur l'IA
        - 📊 Visualisations statistiques
        - 📁 Traitement par batch
        """)
        
        st.info("ℹ️ Utilisez le menu latéral pour naviguer vers les autres sections")
    
    # Inference page
    elif page == "Inférence":
        st.header("Prédiction Simple")
        
        st.write("Entrez les données d'un sinistre pour obtenir une prédiction:")
        
        # Create input form (example structure)
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Données numériques")
            # Add numeric inputs based on your data
            value1 = st.slider("Valeur 1", 0.0, 1.0, 0.5)
            value2 = st.slider("Valeur 2", 0.0, 1.0, 0.5)
        
        with col2:
            st.subheader("Données catégorielles")
            # Add category inputs based on your data
            cat1 = st.selectbox("Catégorie 1", ["Option A", "Option B", "Option C"])
        
        if st.button("🚀 Faire une prédiction"):
            # Prepare input (this is a placeholder)
            input_data = np.array([[value1, value2]])  # Adapt to your model
            
            prediction = model.predict(input_data, verbose=0)
            
            st.success(f"✅ Prédiction: {prediction[0][0]:.4f}")
    
    # Upload page
    elif page == "Upload & Prédiction":
        st.header("Traitement par Batch")
        
        uploaded_file = st.file_uploader("Choisissez un fichier CSV", type=['csv'])
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            
            st.write("**Données chargées:**")
            st.dataframe(df.head())
            
            if st.button("🔄 Traiter et Prédire"):
                with st.spinner("Traitement en cours..."):
                    # Process and predict
                    df_processed = processor.transform_data(df)
                    predictions = model.predict(df_processed.values, verbose=0)
                    
                    # Add predictions to dataframe
                    df['Prédiction'] = predictions
                    
                    st.success("✅ Traitement complété!")
                    st.dataframe(df)
                    
                    # Download option
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Télécharger les résultats",
                        data=csv,
                        file_name="predictions.csv",
                        mime="text/csv"
                    )
    
    # Statistics page
    elif page == "Statistiques":
        st.header("Statistiques du Modèle")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Nombre de paramètres", model.count_params())
        
        with col2:
            st.metric("Couches", len(model.layers))
        
        with col3:
            st.metric("État", "✅ Chargé" if model else "❌ Non chargé")
        
        st.write("**Architecture du modèle:**")
        st.text(str(model.summary()))

if __name__ == '__main__':
    main()
