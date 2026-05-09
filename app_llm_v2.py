"""
Générateur de descriptions de sinistres avec IA
Interface Streamlit pédagogique

Démonstration: Pipeline GenAI complet
- Chargement de données brutes de sinistres
- Transformation et injection dans un LLM
- Génération de descriptions détaillées et professionnelles
- Évaluation de la qualité des résultats
"""
import io
import os
import sys
import time
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
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
LLM_MODEL = 'ollama' 

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
        st.error(f"Erreur LLM: {e}")
        return None

@st.cache_resource
def load_processor():
    """Crée une instance du processeur de données."""
    return DataProcessor()

@st.cache_resource
def load_data():
    """Charge et prétraite les données de sinistres."""
    processor = load_processor()
    df = processor.load_data('Insurance claims data.csv')
    df_clean = processor.clean_data(df)
    # Apply full preprocessing before LLM
    df_processed, _, _ = processor.preprocess_claims(df_clean)
    return df_processed

# ============================================================
# Sidebar - Configuration
# ============================================================
st.sidebar.title("⚙️ Configuration")

page = st.sidebar.radio(
    "Sélectionnez une section",
    ["Accueil", "Analyse des données", "Générer des données", "Générer un résumé de sinistre", "Analyse batch", "À propos"]
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
        <h1>Générateur de sinistres</h1>
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
        - Réduire le temps de traitement
        """)
    
    with col2:
        st.markdown("### Données du projet")
        df = load_data()
        st.metric("Total sinistres", f"{len(df):,}")
        st.metric("Colonnes", df.shape[1])
    
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
        st.write("Production de datasets ou de descriptions précis et détaillés")
    
    st.markdown("---")
    st.info("👉 **Commencez par** Générer des données ou Tester sur un sinistre pour voir le pipeline en action.")
# ============================================================
# PAGE - ANALYSE DES DONNÉES (EDA)
# ============================================================
elif page == "Analyse des données":
    st.markdown("# Analyse Exploratoire des Données (EDA)")
    st.write("Exploration statistique du dataset de sinistres automobiles.")
    
    # 1. Chargement des données (Utilise votre fonction en cache existante)
    df = load_data() 
    
    # Couleurs personnalisées issues de votre notebook
    COLORS = {'sinistre': '#E74C3C', 'non_sinistre': '#2E86AB', 'neutre': '#1F4E79'}
    
    # Colonnes numériques (basé sur votre notebook)
    # Note : Adaptez cette liste si vos colonnes ont des noms légèrement différents dans le df final
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'claim_status' in numeric_cols:
        numeric_cols.remove('claim_status')

    # Séparation par classe
    df0 = df[df['claim_status'] == 0]
    df1 = df[df['claim_status'] == 1]
    
    # Métriques globales
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Polices", f"{len(df):,}")
    col2.metric("Sinistres (1)", f"{len(df1):,}", f"{(len(df1)/len(df)*100):.1f}%")
    col3.metric("Non-sinistres (0)", f"{len(df0):,}", f"{(len(df0)/len(df)*100):.1f}%")
    
    st.markdown("---")
    
    # Organisation en onglets pour la lisibilité
    tab1, tab2, tab3, tab4 = st.tabs(["Statistiques", "Distributions", "Corrélations", "Sélection de variables"])
    
    # ONGLET 1 : STATISTIQUES & DÉSÉQUILIBRE
    with tab1:
        st.subheader("Distribution de la variable cible")
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        counts = df['claim_status'].value_counts()
        
        # Camembert
        axes[0].pie(counts, labels=['Non-sinistre (0)', 'Sinistre (1)'],
                    colors=[COLORS['non_sinistre'], COLORS['sinistre']],
                    autopct='%1.1f%%', startangle=90, wedgeprops={'edgecolor': 'white'})
        axes[0].set_title("Répartition globale")
        
        # Barres
        bars = axes[1].bar(['Non-sinistre (0)', 'Sinistre (1)'], counts.values,
                           color=[COLORS['non_sinistre'], COLORS['sinistre']])
        axes[1].set_title("Effectifs par classe")
        st.pyplot(fig)  # <- Remplace plt.show() dans Streamlit
        
        st.subheader("Statistiques Descriptives")
        desc = df[numeric_cols].describe().T
        desc['skewness'] = df[numeric_cols].skew()
        desc['kurtosis'] = df[numeric_cols].kurtosis()
        st.dataframe(desc[['mean', 'std', 'min', '25%', '50%', '75%', 'max', 'skewness', 'kurtosis']].round(2), use_container_width=True)

    # ONGLET 2 : DISTRIBUTIONS
    with tab2:
        st.subheader("Distributions des variables clés")
        # Sélection de quelques variables clés présentes dans votre set
        key_vars = [col for col in ['customer_age', 'vehicle_age', 'region_density'] if col in df.columns]
        
        if key_vars:
            fig, axes = plt.subplots(1, len(key_vars), figsize=(5*len(key_vars), 4))
            if len(key_vars) == 1: axes = [axes] # Sécurité si une seule variable
            
            for i, col in enumerate(key_vars):
                axes[i].hist(df1[col].dropna(), bins=30, alpha=0.6, color=COLORS['sinistre'], label='Sinistre', density=True)
                axes[i].hist(df0[col].dropna(), bins=30, alpha=0.4, color=COLORS['non_sinistre'], label='Non-sinistre', density=True)
                axes[i].set_title(col)
                axes[i].legend()
            st.pyplot(fig)
            
        st.info("💡 L'analyse révèle des valeurs aberrantes naturelles (outliers) comme des véhicules très anciens. Ils sont conservés car pertinents actuariellement.")

    # ONGLET 3 : CORRÉLATIONS
    with tab3:
        st.subheader("Matrice de corrélations")
        corr_matrix = df[numeric_cols + ['claim_status']].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='RdBu_r', center=0, vmin=-1, vmax=1, ax=ax)
        st.pyplot(fig)
        
        st.subheader("Corrélation avec les sinistres (claim_status)")
        corr_target = corr_matrix['claim_status'].drop('claim_status').abs().sort_values(ascending=True)
        
        fig, ax = plt.subplots(figsize=(8, max(4, len(corr_target)*0.3)))
        colors_bar = [COLORS['sinistre'] if v > 0.05 else COLORS['non_sinistre'] for v in corr_target]
        ax.barh(corr_target.index, corr_target.values, color=colors_bar)
        ax.axvline(x=0.05, color='gray', linestyle='--')
        ax.set_xlabel("Corrélation absolue")
        st.pyplot(fig)

    # ONGLET 4 : SÉLECTION DE VARIABLES
    with tab4:
        st.subheader("Pertinence des variables (Pearson & Spearman)")
        
        # Calcul Pearson et Spearman
        corr_pearson = df[numeric_cols].corrwith(df['claim_status']).abs()
        
        spearman_data = []
        for col in numeric_cols:
            rho, pval = stats.spearmanr(df[col].fillna(0), df['claim_status'])
            spearman_data.append({'Variable': col, 'Pearson_abs': corr_pearson[col], 'Spearman_rho': abs(rho), 'P_value': pval})
            
        df_selection = pd.DataFrame(spearman_data).sort_values('Pearson_abs', ascending=False)
        df_selection['Significatif'] = df_selection['P_value'] < 0.05
        
        st.dataframe(df_selection.style.highlight_max(subset=['Pearson_abs', 'Spearman_rho'], color='lightgreen'), use_container_width=True)
        
        st.success("""
        **Conclusions de l'EDA :**
        - Faibles corrélations linéaires globales (<0.10) indiquant des relations complexes.
        - Le déséquilibre sévère (1 sinistre pour 14 non-sinistres) justifie notre approche de **génération de données synthétiques** par IA pour rééquilibrer le dataset.
        """)
# ============================================================
# NOUVELLE PAGE - GÉNÉRER DES DONNÉES SYNTHÉTIQUES
# ============================================================
elif page == "Générer des données":
    st.markdown("# Génération de données synthétiques")
    st.write("""
    Créez un jeu de données de sinistres totalement inventé par l'IA, 
    mais qui respecte statistiquement le schéma de la base de données réelle.
    """)
    
    generator = load_generator()
    if generator is None:
        st.stop()
        
    df_original = load_data()
    processor = load_processor()
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Configuration")
        n_rows = st.number_input("Nombre de lignes à générer", min_value=1, max_value=50, value=5)
        generate_btn = st.button("🚀 Générer le CSV", type="primary", use_container_width=True)
        
    with col2:
        st.info(f"Le modèle va se baser sur les {len(df_original.columns)} colonnes de votre jeu de données actuel pour inventer de nouveaux profils cohérents.")

    if generate_btn:
        with st.spinner(f"⏳ Génération de {n_rows} lignes par le LLM en cours..."):
            start = time.time()
            
            # 1. On filtre pour ne garder QUE les sinistres réels
            df_sinistres = df_original[df_original['claim_status'] == 1]
            
            # 2. Le schéma extrait sera donc mathématiquement exact pour les sinistres
            schema_info = processor.get_schema_summary(df_sinistres)
            
            # 3. On tire 4 vrais exemples au hasard parmi les sinistres
            dynamic_examples = processor.get_dynamic_examples(df_sinistres, n=4)
            # ------------------------
            
            # 4. Appeler le LLM avec ces nouvelles informations
            raw_csv_output = generator.generate_synthetic_data(schema_info, dynamic_examples, n_rows)
            
            clean_csv = raw_csv_output
            if clean_csv.startswith("```"):
                clean_csv = "\n".join(clean_csv.split("\n")[1:-1])
                
            elapsed = time.time() - start
            
            try:
                df_synth = pd.read_csv(io.StringIO(clean_csv))
                st.success(f"✅ Génération de profils 'Sinistrés' réussie en {elapsed:.1f}s !")
                st.dataframe(df_synth, use_container_width=True)
                
                st.download_button(
                    label="📥 Télécharger ce dataset au format CSV",
                    data=clean_csv,
                    file_name=f"sinistres_synthetiques_ia_{n_rows}_lignes.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            except Exception as e:
                st.error("Le LLM a échoué à générer un CSV. Réponse brute :")
                st.code(raw_csv_output, language="text")
                st.exception(e)

# ============================================================
# PAGE 2 - TESTER SUR UN SINISTRE
# ============================================================
elif page == "Générer un résumé de sinistre":
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
        # Format data with DataProcessor before sending to LLM
        processor = load_processor()
        formatted_claim = processor.format_for_llm(claim_data)
        
        with st.spinner("⏳ Génération en cours..."):
            start = time.time()
            description = generator.generate(formatted_claim)
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
        processor = load_processor()
        
        start_time = time.time()
        
        for i, claim in enumerate(claims_data):
            iter_start = time.time()
            
            # Format data before sending to LLM
            formatted_claim = processor.format_for_llm(claim)
            description = generator.generate(formatted_claim, max_length=300)
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
elif page == "À propos":
    st.markdown("# À propos du projet")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("## 🎓 Contexte académique")
        st.write("""
        **Projet M2 ES - ISFA**
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
    **Modèle**: phi 3.5 (via Ollama)
    
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
