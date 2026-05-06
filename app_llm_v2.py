"""
Générateur de descriptions de sinistres avec IA
Interface Streamlit pédagogique

Démonstration: Pipeline GenAI complet
- Chargement de données brutes de sinistres
- Transformation et injection dans un LLM
- Génération de descriptions détaillées et professionnelles
- Évaluation de la qualité des résultats
"""

import os
import sys
import time
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

# Add src to path
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

from data_processor import DataProcessor
from model import create_generator

# ============================================================
# Configuration Streamlit
# ============================================================
st.set_page_config(
    page_title="Sinistres IA",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
    <style>
    .title-container {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        color: white;
        margin-bottom: 20px;
    }
    .metric-box {
        padding: 15px;
        background-color: #f0f2f6;
        border-radius: 8px;
        border-left: 4px solid #667eea;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# Configuration LLM
# ============================================================
LLM_MODEL = 'ollama'  # Ollama only - local, free LLM

if LLM_MODEL == 'ollama':
    import requests
    try:
        requests.get("http://localhost:11434/api/tags", timeout=2)
    except:
        st.error("""
         **Ollama n'est pas accessible!**
        
        Lancez Ollama dans un terminal:
        ```bash
        ollama serve
        ```
        
        Et dans un autre terminal:
        ```bash
        ollama pull mistral
        ```
        """)
        st.stop()

# ============================================================
# Cache des ressources
# ============================================================
@st.cache_resource
def load_generator():
    """Charge le générateur LLM."""
    try:
        generator = create_generator(model_name=LLM_MODEL)
        return generator
    except Exception as e:
        st.error(f"❌ Erreur LLM: {e}")
        return None

@st.cache_resource
def load_data():
    """Charge les données de sinistres."""
    processor = DataProcessor()
    df = processor.load_data('Insurance claims data.csv')
    df_clean = processor.clean_data(df)
    return df_clean

# ============================================================
# Sidebar - Configuration
# ============================================================
st.sidebar.title("⚙️ Configuration")

page = st.sidebar.radio(
    "Sélectionnez une section",
    ["Accueil", "Tester sur un sinistre", "Analyse batch", "À propos"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
### À propos du projet
**Génération de descriptions de sinistres avec IA**

Projet M2 ISFA - Extraction et valorisation de données

**Technologies:**
- LLM: orca-mini (via Ollama)
- Données: 58K sinistres auto
- Stack: Python, Streamlit
""")

# ============================================================
# PAGE 1 - ACCUEIL
# ============================================================
if page == "Accueil":
    st.markdown("""
    <div class="title-container">
        <h1>Générateur de descriptions de sinistres</h1>
        <p><i>Intelligence Artificielle appliquée à l'assurance automobile</i></p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Objectif du projet")
        st.write("""
        Ce projet démontre comment les **LLMs (Large Language Models)** peuvent 
        automatiser la génération de descriptions professionnelles pour sinistres 
        d'assurance automobile.
        
        **Cas d'usage:**
        - Automatiser la rédaction de rapports
        - Augmenter la productivité des experts
        - Standardiser la qualité des descriptions
        - ⏱Réduire le temps de traitement
        """)
    
    with col2:
        st.markdown("### Données du projet")
        df = load_data()
        st.metric("Total sinistres", f"{len(df):,}")
        st.metric("Colonnes", df.shape[1])
        st.metric("Véhicules uniques", df['model'].nunique())
        st.metric("Régions couvertes", df['region_code'].nunique())
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 1️⃣ Ingestion")
        st.write("Chargement de 58K sinistres structurés")
    
    with col2:
        st.markdown("### 2️⃣ Transformation")
        st.write("Création de prompts optimisés pour l'IA")
    
    with col3:
        st.markdown("### 3️⃣ Génération")
        st.write("Production de descriptions précises et détaillées")
    
    st.markdown("---")
    st.info("👉 **Commencez par** Tester sur un sinistre pour voir le pipeline en action.")

# ============================================================
# PAGE 2 - TESTER SUR UN SINISTRE
# ============================================================
elif page == "Tester sur un sinistre":
    st.markdown("# Génération interactive")
    st.write("Sélectionnez ou créez un sinistre pour générer une description")
    
    generator = load_generator()
    if generator is None:
        st.stop()
    
    df = load_data()
    
    # Options: Sélectionner depuis BD ou créer manuellement
    mode = st.radio("Mode", ["Sélectionner depuis la BD", "Saisir manuellement"])
    
    if mode == "Sélectionner depuis la BD":
        idx = st.slider("Sélectionnez un sinistre", 0, min(len(df)-1, 100), 0)
        claim_data = df.iloc[idx].to_dict()
    else:
        st.subheader("Saisir les données du sinistre")
        col1, col2 = st.columns(2)
        
        with col1:
            claim_data = {
                "vehicle_age": st.number_input("Âge véhicule (ans)", 0.0, 30.0, 2.0),
                "customer_age": st.number_input("Âge client (ans)", 18, 100, 40),
                "fuel_type": st.selectbox("Carburant", ["Petrol", "Diesel", "CNG", "Hybrid"]),
                "transmission_type": st.selectbox("Transmission", ["Manual", "Automatic"]),
            }
        
        with col2:
            claim_data.update({
                "airbags": st.number_input("Airbags", 0, 12, 2),
                "ncap_rating": st.number_input("Notation NCAP", 0, 5, 3),
                "segment": st.selectbox("Segment", ["A", "B", "C", "D"]),
                "is_esc": st.selectbox("ESC", ["Yes", "No"]),
            })
    
    # Afficher les données
    st.markdown("### Données du sinistre")
    st.json(claim_data)
    
    # Bouton Générer
    if st.button("Générer description", type="primary", use_container_width=True):
        with st.spinner("⏳ Génération en cours..."):
            start = time.time()
            description = generator.generate(claim_data)
            elapsed = time.time() - start
        
        st.success(f"✅ Génération terminée en {elapsed:.1f}s")
        
        # Afficher la description
        st.markdown("### Description générée")
        st.write(description)
        
        # Métriques
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("⏱️ Temps", f"{elapsed:.1f}s")
        with col2:
            words = len(description.split())
            st.metric("📝 Mots", words)
        with col3:
            chars = len(description)
            st.metric("🔤 Caractères", chars)
        
        # Bouton Copier
        st.code(description, language="text")

# ============================================================
# PAGE 3 - ANALYSE BATCH
# ============================================================
elif page == "Analyse batch":
    st.markdown("# Traitement par batch")
    
    generator = load_generator()
    if generator is None:
        st.stop()
    
    df = load_data()
    
    # Sélectionner nombre de sinistres
    n_claims = st.slider("Nombre de sinistres à traiter", 1, 50, 5)
    
    if st.button("Lancer le traitement", type="primary", use_container_width=True):
        claims_data = df.head(n_claims).to_dict('records')
        
        progress_bar = st.progress(0)
        results = []
        
        start_time = time.time()
        
        for i, claim in enumerate(claims_data):
            iter_start = time.time()
            
            description = generator.generate(claim, max_length=300)
            iter_time = time.time() - iter_start
            
            results.append({
                "ID Sinistre": claim.get('policy_id', f'CLAIM_{i}'),
                "Véhicule": claim.get('model', 'N/A'),
                "Âge Client": claim.get('customer_age', 'N/A'),
                "Description": description[:100] + "..." if len(description) > 100 else description,
                "Temps (s)": round(iter_time, 2),
                "Mots": len(description.split()),
            })
            
            progress_bar.progress((i + 1) / n_claims)
        
        total_time = time.time() - start_time
        
        st.success(f"✅ {n_claims} sinistres traités en {total_time:.1f}s")
        
        # Tableau résultats
        st.markdown("### Résultats")
        results_df = pd.DataFrame(results)
        st.dataframe(results_df, use_container_width=True)
        
        # Statistiques
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("⏱️ Temps moyen", f"{total_time/n_claims:.1f}s/sinistre")
        with col2:
            st.metric("📝 Mots moyen", int(results_df['Mots'].mean()))
        with col3:
            st.metric("🏃 Throughput", f"{n_claims/total_time:.1f} sinistres/min")
        with col4:
            st.metric("✅ Succès", f"{len(results)}/{n_claims}")
        
        # Télécharger résultats
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Télécharger résultats (CSV)",
            data=csv,
            file_name=f"sinistres_generes_{int(time.time())}.csv",
            mime="text/csv",
        )

# ============================================================
# PAGE 4 - À PROPOS
# ============================================================
elif page == "📚 À propos":
    st.markdown("# À propos du projet")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("## 🎓 Contexte académique")
        st.write("""
        **Projet M2 - ISFA**
        Celia IMAKHLOUFEN (DSI), Fatoumata Binta DIALLO (DARM), Siderlin MOUPOLO (EQUADE)
        """)
        
        st.markdown("## 🛠️ Architecture technique")
        st.write("""
        ```
        ┌─ Données brutes (CSV)
        │       ↓
        ├─ Preprocessing & cleaning
        │       ↓
        ├─ Création de prompts structurés
        │       ↓
        ├─ LLM (via Ollama)
        │       ↓
        └─ Descriptions générées
        ```
        """)
    
    with col2:
        st.markdown("## Statistiques clés")
        df = load_data()
        
        stats = {
            "Total sinistres": len(df),
            "Colonnes features": df.shape[1],
            "Âge client (moyen)": f"{df['customer_age'].mean():.0f} ans",
            "Âge véhicule (moyen)": f"{df['vehicle_age'].mean():.1f} ans",
            "Carburants": df['fuel_type'].nunique(),
            "Segments": df['segment'].nunique(),
        }
        
        for key, value in stats.items():
            st.write(f"- **{key}**: {value}")
    
    st.markdown("---")
    
    st.markdown("## 🤖 Modèle IA utilisé")
    st.write(f"""
    **Modèle**: Mistral-7B (via Ollama)
    
    - Modèle open-source, licence Apache 2.0
    - 7 milliards de paramètres
    - Exécution locale (confidentialité garantie)
    - Compatible API OpenAI
    - Temps de réponse: ~10-20s/description
    """)
    
    st.markdown("## 💡 Améliorations possibles")
    st.write("""
    - Fine-tuning du modèle sur corpus de sinistres
    - Évaluation de qualité avec LLM-as-Judge
    - Stockage des descriptions en base de données
    - API REST pour intégration
    - Traitement parallélisé pour batch
    - Caching des embeddings pour optimisation
    """)
