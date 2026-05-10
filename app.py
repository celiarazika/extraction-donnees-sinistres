"""
Application Streamlit — Génération de données synthétiques de sinistres
Projet ISFA 2025-2026
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import f1_score, recall_score, precision_score

# ── Configuration ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sinistres Synthétiques",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

COLORS = {'sinistre': '#E74C3C', 'non_sinistre': '#2E86AB',
          'neutre': '#1F4E79', 'accent': '#2ECC71',
          'ctgan': '#2ECC71', 'tvae': '#F39C12'}

st.markdown("""
<style>
    .main-title { font-size:2.2rem; font-weight:bold; color:#1F4E79; text-align:center; padding:1rem 0 0.5rem 0; }
    .sub-title { font-size:1rem; color:#666; text-align:center; margin-bottom:2rem; }
    .metric-card { background:#F5F8FC; border-radius:10px; padding:1rem; text-align:center; border-left:4px solid #1F4E79; }
    .metric-value { font-size:1.8rem; font-weight:bold; color:#1F4E79; }
    .metric-label { font-size:0.85rem; color:#666; margin-top:0.2rem; }
    .section-header { font-size:1.3rem; font-weight:bold; color:#2E75B6; border-bottom:2px solid #2E75B6; padding-bottom:0.3rem; margin:1.5rem 0 1rem 0; }
    .info-box { background:#EBF5FB; border-radius:8px; padding:0.8rem 1rem; border-left:4px solid #2E86AB; margin:0.5rem 0; }
    .warning-box { background:#FEF9E7; border-radius:8px; padding:0.8rem 1rem; border-left:4px solid #F39C12; margin:0.5rem 0; }
    .success-box { background:#EAFAF1; border-radius:8px; padding:0.8rem 1rem; border-left:4px solid #2ECC71; margin:0.5rem 0; }
    .danger-box { background:#FDEDEC; border-radius:8px; padding:0.8rem 1rem; border-left:4px solid #E74C3C; margin:0.5rem 0; }
    .llm-box { background:#F8F4FF; border-radius:8px; padding:1rem; border-left:4px solid #8E44AD; margin:0.5rem 0; font-style:italic; }
</style>
""", unsafe_allow_html=True)

# ── Chargement ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    try: return pd.read_csv('outputs/data_encoded.csv')
    except: return None

@st.cache_data
def load_raw():
    try: return pd.read_csv('data/Insurance claims data.csv', sep=',')
    except: return None

@st.cache_data
def load_results():
    try: return pd.read_csv('outputs/resultats_modelisation.csv')
    except: return None

@st.cache_data
def load_eval_supervisee():
    try: return pd.read_csv('outputs/evaluation_supervisee.csv')
    except: return None

@st.cache_data
def load_ks():
    try: return pd.read_csv('outputs/ks_test_results.csv')
    except: return None

@st.cache_data
def load_sensibilite_volume():
    try: return pd.read_csv('outputs/sensibilite_volume.csv')
    except: return None

@st.cache_data
def load_preprocessed():
    try: return pd.read_csv('outputs/data_preprocessed.csv')
    except: return None

@st.cache_data
def load_synthetic(generateur):
    try: return pd.read_csv(f'outputs/synthetic/synthetic_{generateur.lower()}.csv')
    except: return None

@st.cache_data
def load_llm():
    try: return pd.read_csv('outputs/llm/sinistres_avec_descriptions.csv')
    except: return None

@st.cache_resource
def load_model(path):
    try:
        with open(path, 'rb') as f: return pickle.load(f)
    except: return None

df       = load_data()
df_raw   = load_raw()
df_pre   = load_preprocessed()
results  = load_results()
eval_sup = load_eval_supervisee()
df_ks    = load_ks()
df_vol   = load_sensibilite_volume()
df_llm   = load_llm()

try:
    with open('outputs/models/model_info.json') as f: info = json.load(f)
except: info = None

data_ok  = df is not None
model_ok = results is not None
eval_ok  = eval_sup is not None
ctgan_ok = os.path.exists('outputs/synthetic/synthetic_ctgan.csv')
tvae_ok  = os.path.exists('outputs/synthetic/synthetic_tvae.csv')
llm_ok   = df_llm is not None

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚗 Navigation")
    page = st.radio("Menu", [
        "🏠 Accueil",
        "📊 Exploration des données",
        "🤖 Génération synthétique",
        "🧠 Descriptions LLM",
        "📈 Évaluation du modèle",
        "🔮 Prédiction individuelle",
        "⚠️ Limites et défis"
    ])
    st.markdown("---")
    st.markdown("**Projet ISFA 2025-2026**")
    st.markdown("Génération de données synthétiques de sinistres")
    st.markdown("---")
    if data_ok: st.success(f"✅ Dataset : {df.shape[0]:,} polices")
    else: st.error("❌ Dataset non chargé")
    if model_ok: st.success("✅ Modèles chargés")
    else: st.warning("⚠️ Lancer etape3_modelisation.py")
    if llm_ok: st.success("✅ Descriptions LLM disponibles")
    else: st.warning("⚠️ Lancer etape3_modelisation.py")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — ACCUEIL
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Accueil":
    st.markdown('<div class="main-title">🚗 Génération de Données Synthétiques de Sinistres</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Projet IA & Assurance — ISFA 2025-2026</div>', unsafe_allow_html=True)
    st.markdown("---")

    if data_ok:
        col1, col2, col3, col4 = st.columns(4)
        n_total = len(df); n_claims = int(df['claim_status'].sum()); n_no = n_total - n_claims
        with col1: st.markdown(f'<div class="metric-card"><div class="metric-value">{n_total:,}</div><div class="metric-label">Polices total</div></div>', unsafe_allow_html=True)
        with col2: st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#E74C3C">{n_claims:,}</div><div class="metric-label">Sinistres (6.4%)</div></div>', unsafe_allow_html=True)
        with col3: st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#2E86AB">{n_no:,}</div><div class="metric-label">Non-sinistres (93.6%)</div></div>', unsafe_allow_html=True)
        with col4: st.markdown(f'<div class="metric-card"><div class="metric-value">1 / 14</div><div class="metric-label">Ratio d\'imbalance</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown('<div class="section-header">📌 Problématique</div>', unsafe_allow_html=True)
        st.markdown("""
        Les données de sinistres sont naturellement **déséquilibrées** : seulement **6,4%** des polices génèrent un sinistre.

        **Solution :** Deux approches complémentaires :
        - **CTGAN/TVAE** → génération de données tabulaires synthétiques
        - **Ollama/phi3.5** → génération de descriptions textuelles professionnelles
        """)
    with col_right:
        st.markdown('<div class="section-header">🗂️ Pipeline</div>', unsafe_allow_html=True)
        st.markdown("""
        | Étape | Description | Statut |
        |-------|-------------|--------|
        | 1 | Data Acquisition & Prétraitement | ✅ |
        | 2 | Analyse Exploratoire (EDA) | ✅ |
        | 3 | Modélisation CTGAN/TVAE + LLM | ✅ |
        | 4 | Évaluation & Sensibilité | ✅ |
        | 5 | Limites & Perspectives | ⏳ |
        """)

    if model_ok:
        st.markdown("---")
        st.markdown('<div class="section-header">🏆 Résultats clés</div>', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("Meilleur AUC-ROC", f"{results['auc'].max():.4f}", results.loc[results['auc'].idxmax(),'model'])
        with col2: st.metric("Meilleur F1", f"{results['f1'].max():.4f}", results.loc[results['f1'].idxmax(),'model'])
        with col3: st.metric("Meilleur Recall", f"{results['recall'].max():.4f}", results.loc[results['recall'].idxmax(),'model'])
        with col4: st.metric("Synthétiques générés", "10 000", "5K CTGAN + 5K TVAE")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — EXPLORATION DES DONNÉES
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Exploration des données":
    st.markdown('<div class="main-title">📊 Exploration des données</div>', unsafe_allow_html=True)
    if not data_ok: st.error("Dataset non disponible."); st.stop()

    df0 = df[df['claim_status']==0]; df1 = df[df['claim_status']==1]
    numeric_cols = ['subscription_length','vehicle_age','customer_age','region_density',
                    'displacement','cylinder','turning_radius','length','width',
                    'gross_weight','torque_nm','torque_rpm','power_bhp','power_rpm','airbags','ncap_rating']

    tab1, tab2, tab3, tab4 = st.tabs(["📋 Statistiques","📈 Distributions","🔗 Corrélations","🏷️ Catégorielles"])

    with tab1:
        desc = df[numeric_cols].describe().T.round(2)
        desc['skewness'] = df[numeric_cols].skew().round(2)
        desc['kurtosis'] = df[numeric_cols].kurtosis().round(2)
        st.dataframe(desc[['mean','std','min','25%','50%','75%','max','skewness','kurtosis']], use_container_width=True)
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(5,5))
            counts = df['claim_status'].value_counts()
            ax.pie(counts, labels=['Non-sinistre','Sinistre'],
                   colors=[COLORS['non_sinistre'],COLORS['sinistre']],
                   autopct='%1.1f%%', startangle=90, wedgeprops={'edgecolor':'white','linewidth':2})
            ax.set_title("Répartition des classes", fontweight='bold')
            st.pyplot(fig); plt.close()
        with col2:
            st.metric("Sinistres", f"{int(df['claim_status'].sum()):,}", "6.4%")
            st.metric("Non-sinistres", f"{int((df['claim_status']==0).sum()):,}", "93.6%")
            st.markdown('<div class="warning-box">⚠️ 1 sinistre pour 14 non-sinistres</div>', unsafe_allow_html=True)

    with tab2:
        col_sel = st.selectbox("Variable", numeric_cols)
        fig, ax = plt.subplots(figsize=(10,4))
        ax.hist(df0[col_sel], bins=40, alpha=0.5, color=COLORS['non_sinistre'], label='Non-sinistre', density=True)
        ax.hist(df1[col_sel], bins=40, alpha=0.7, color=COLORS['sinistre'], label='Sinistre', density=True)
        ax.set_title(f"Distribution de {col_sel}", fontweight='bold'); ax.legend()
        st.pyplot(fig); plt.close()

    with tab3:
        corr_cols = st.multiselect("Variables", numeric_cols+['claim_status'],
                                    default=['subscription_length','vehicle_age','customer_age','region_density','cylinder','claim_status'])
        if len(corr_cols) >= 2:
            fig, ax = plt.subplots(figsize=(10,7))
            corr_m = df[corr_cols].corr()
            mask = np.triu(np.ones_like(corr_m, dtype=bool))
            sns.heatmap(corr_m, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=ax)
            st.pyplot(fig); plt.close()

    with tab4:
        if df_raw is not None:
            cat_var = st.selectbox("Variable catégorielle", ['fuel_type','segment','transmission_type','rear_brakes_type','steering_type'])
            taux = df_raw.groupby(cat_var)['claim_status'].mean().sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(8,4))
            bars = ax.bar(taux.index, taux.values*100, color=COLORS['sinistre'], alpha=0.7, edgecolor='white')
            ax.axhline(y=6.4, color='gray', linestyle='--', linewidth=1.5, label='Moyenne (6.4%)')
            ax.set_ylabel("Taux (%)"); ax.set_title(f"Taux de sinistres par {cat_var}", fontweight='bold')
            ax.tick_params(axis='x', rotation=30); ax.legend()
            for bar, val in zip(bars, taux.values):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05, f'{val*100:.1f}%', ha='center', fontsize=9)
            st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — GÉNÉRATION SYNTHÉTIQUE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Génération synthétique":
    st.markdown('<div class="main-title">🤖 Génération de données synthétiques</div>', unsafe_allow_html=True)

    if ctgan_ok or tvae_ok:
        st.markdown('<div class="success-box">✅ <strong>Données synthétiques disponibles</strong> — CTGAN et TVAE entraînés</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="warning-box">⏳ Exécuter etape3_modelisation.py</div>', unsafe_allow_html=True)
        st.stop()

    st.markdown("---")

    # Sélecteur générateur
    options = []
    if ctgan_ok: options.append("CTGAN")
    if tvae_ok:  options.append("TVAE")
    if ctgan_ok and tvae_ok: options.append("Comparer CTGAN vs TVAE")

    choix = st.radio("🔍 Générateur à explorer", options, horizontal=True)

    # Chargement des données selon le choix
    if choix in ["CTGAN", "Comparer CTGAN vs TVAE"]:
        syn_ctgan = load_synthetic("ctgan")
    if choix in ["TVAE", "Comparer CTGAN vs TVAE"]:
        syn_tvae = load_synthetic("tvae")

    df_sinistres = df[df['claim_status']==1] if data_ok else None

    st.markdown("---")

    if choix == "CTGAN":
        synthetic = syn_ctgan
        color     = COLORS['ctgan']
        label     = "CTGAN"
    elif choix == "TVAE":
        synthetic = syn_tvae
        color     = COLORS['tvae']
        label     = "TVAE"

    if choix != "Comparer CTGAN vs TVAE":
        # Métriques
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("Sinistres synthétiques", f"{len(synthetic):,}")
        with col2: st.metric("Colonnes", f"{synthetic.shape[1]}")
        with col3: st.metric("Sinistres réels (train)", "2 998")
        with col4: st.metric("Ratio augmentation", f"{len(synthetic)/2998:.1f}x")

        st.markdown("---")
        tab1, tab2, tab3 = st.tabs(["📋 Tableau des données", "📈 Distributions", "📊 Statistiques comparées"])

        with tab1:
            st.markdown(f'<div class="section-header">Les {len(synthetic):,} sinistres synthétiques {label}</div>', unsafe_allow_html=True)
            # Filtres
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                n_show = st.slider("Nombre de lignes à afficher", 10, min(500, len(synthetic)), 50)
            with col_f2:
                if 'subscription_length' in synthetic.columns:
                    sl_min = float(synthetic['subscription_length'].min())
                    sl_max = float(synthetic['subscription_length'].max())
                    sl_range = st.slider("Filtrer par subscription_length", sl_min, sl_max, (sl_min, sl_max))
                    filtered = synthetic[(synthetic['subscription_length'] >= sl_range[0]) &
                                        (synthetic['subscription_length'] <= sl_range[1])]
                else:
                    filtered = synthetic

            st.dataframe(filtered.head(n_show), use_container_width=True)
            st.download_button(
                label=f"⬇️ Télécharger les {len(synthetic):,} sinistres {label}",
                data=synthetic.to_csv(index=False).encode('utf-8'),
                file_name=f"synthetic_{label.lower()}.csv",
                mime='text/csv'
            )

        with tab2:
            numeric_key = ['subscription_length','vehicle_age','customer_age',
                           'region_density','power_bhp','torque_nm']
            available = [c for c in numeric_key if c in synthetic.columns]
            var = st.selectbox("Variable", available)

            fig, axes = plt.subplots(1, 2, figsize=(12,4))
            if df_sinistres is not None and var in df_sinistres.columns:
                axes[0].hist(df_sinistres[var].dropna(), bins=30, color=COLORS['sinistre'], alpha=0.8)
                axes[0].set_title(f"{var} — Réels (2 998)", fontweight='bold')
            axes[1].hist(synthetic[var].dropna(), bins=30, color=color, alpha=0.8)
            axes[1].set_title(f"{var} — {label} synthétiques ({len(synthetic):,})", fontweight='bold')
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with tab3:
            cols_stat = [c for c in numeric_key if c in synthetic.columns and
                         (df_sinistres is not None and c in df_sinistres.columns)]
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Sinistres réels (2 998)**")
                if df_sinistres is not None:
                    st.dataframe(df_sinistres[cols_stat].describe().round(3), use_container_width=True)
            with col2:
                st.write(f"**Synthétiques {label} ({len(synthetic):,})**")
                st.dataframe(synthetic[cols_stat].describe().round(3), use_container_width=True)

    else:
        # Comparaison CTGAN vs TVAE
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Sinistres réels", "2 998")
        with col2: st.metric("Synthétiques CTGAN", f"{len(syn_ctgan):,}")
        with col3: st.metric("Synthétiques TVAE", f"{len(syn_tvae):,}")

        st.markdown("---")
        numeric_key = ['subscription_length','vehicle_age','customer_age','region_density']
        available = [c for c in numeric_key if c in syn_ctgan.columns and c in syn_tvae.columns]
        var = st.selectbox("Variable à comparer", available)

        fig, axes = plt.subplots(1, 3, figsize=(16,4))
        if df_sinistres is not None and var in df_sinistres.columns:
            axes[0].hist(df_sinistres[var].dropna(), bins=30, color=COLORS['sinistre'], alpha=0.8)
            axes[0].set_title(f"{var} — Réels (2 998)", fontweight='bold')
        axes[1].hist(syn_ctgan[var].dropna(), bins=30, color=COLORS['ctgan'], alpha=0.8)
        axes[1].set_title(f"{var} — CTGAN ({len(syn_ctgan):,})", fontweight='bold')
        axes[2].hist(syn_tvae[var].dropna(), bins=30, color=COLORS['tvae'], alpha=0.8)
        axes[2].set_title(f"{var} — TVAE ({len(syn_tvae):,})", fontweight='bold')
        plt.tight_layout(); st.pyplot(fig); plt.close()

        # Téléchargements
        col1, col2 = st.columns(2)
        with col1:
            st.download_button("⬇️ Télécharger CTGAN", syn_ctgan.to_csv(index=False).encode('utf-8'),
                               "synthetic_ctgan.csv", "text/csv")
        with col2:
            st.download_button("⬇️ Télécharger TVAE", syn_tvae.to_csv(index=False).encode('utf-8'),
                               "synthetic_tvae.csv", "text/csv")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — DESCRIPTIONS LLM
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🧠 Descriptions LLM":
    st.markdown('<div class="main-title">🧠 Descriptions LLM — Ollama/phi3.5</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">Architecture LLM</div>', unsafe_allow_html=True)
        st.markdown("""
        **Ollama** — serveur LLM local (port 11434)
        - Gratuit, sans API key, données en local
        - API compatible OpenAI

        **phi3.5** — modèle Microsoft
        - 2.2 GB téléchargé localement
        - Température 0.2 — réponses précises
        - 500 tokens maximum par description
        """)
    with col2:
        st.markdown('<div class="section-header">Objectif</div>', unsafe_allow_html=True)
        st.markdown("""
        Enrichir les dossiers sinistres avec une **description textuelle professionnelle** générée automatiquement.

        Pour chaque sinistre (réel ou synthétique), le LLM rédige un paragraphe de 50 à 100 mots comme un expert en assurance le ferait.
        """)

    st.markdown("---")

    if llm_ok:
        st.markdown('<div class="success-box">✅ Descriptions LLM disponibles — 10 descriptions générées</div>', unsafe_allow_html=True)
        st.markdown("---")

        # Navigation entre les descriptions
        st.markdown('<div class="section-header">Explorer les descriptions générées</div>', unsafe_allow_html=True)

        idx = st.slider("Sinistre", 1, len(df_llm), 1) - 1
        row = df_llm.iloc[idx]

        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("**Données du sinistre :**")
            data_cols = [c for c in df_llm.columns if c != 'description_llm']
            for col in data_cols:
                st.markdown(f"- **{col}** : {row[col]}")

        with col2:
            st.markdown("**Description générée par phi3.5 :**")
            st.markdown(f'<div class="llm-box">📝 {row["description_llm"]}</div>', unsafe_allow_html=True)

        st.markdown("---")

        # Tableau complet
        st.markdown('<div class="section-header">Toutes les descriptions</div>', unsafe_allow_html=True)
        st.dataframe(df_llm, use_container_width=True)

        st.download_button(
            label="⬇️ Télécharger les descriptions LLM",
            data=df_llm.to_csv(index=False).encode('utf-8'),
            file_name="sinistres_avec_descriptions.csv",
            mime='text/csv'
        )

    else:
        st.markdown('<div class="warning-box">⚠️ Descriptions LLM non disponibles — Exécuter etape3_modelisation.py avec Ollama actif</div>', unsafe_allow_html=True)
        st.markdown("""
        **Pour activer Ollama :**
        1. Installer Ollama : https://ollama.com
        2. Télécharger le modèle : `ollama pull phi3.5`
        3. Lancer le serveur : `ollama serve`
        4. Relancer le script : `python etape3_modelisation.py`
        """)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — ÉVALUATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Évaluation du modèle":
    st.markdown('<div class="main-title">📈 Évaluation du modèle</div>', unsafe_allow_html=True)
    if not model_ok: st.error("Résultats non disponibles."); st.stop()
    st.markdown('<div class="success-box">✅ Étapes 3 et 4 terminées</div>', unsafe_allow_html=True)
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["📊 Supervisée","🔬 Non supervisée","📉 Sensibilité"])

    with tab1:
        st.markdown('<div class="section-header">Métriques supervisées</div>', unsafe_allow_html=True)
        df_show = eval_sup if eval_ok else results
        st.dataframe(df_show.round(4), use_container_width=True)

        metric_choice = st.selectbox("Métrique", ['auc','f1','recall','precision'])
        fig, ax = plt.subplots(figsize=(10,5))
        color_map = {'auc': COLORS['neutre'], 'f1': COLORS['sinistre'],
                     'recall': COLORS['ctgan'], 'precision': COLORS['tvae']}
        vals = df_show[metric_choice]
        bars = ax.bar(df_show['model'], vals, color=color_map[metric_choice], alpha=0.8, edgecolor='white')
        ax.set_title(f"{metric_choice.upper()} par modèle", fontweight='bold')
        ax.tick_params(axis='x', rotation=20)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
                    f'{val:.4f}', ha='center', fontsize=10, fontweight='bold')
        plt.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown("---")
        st.markdown('<div class="section-header">Courbes ROC</div>', unsafe_allow_html=True)
        model_files = {
            'XGBoost Baseline': 'outputs/models/xgb_baseline.pkl',
            'XGBoost + CTGAN' : 'outputs/models/xgb_ctgan.pkl',
            'XGBoost + TVAE'  : 'outputs/models/xgb_tvae.pkl',
            'XGBoost + SMOTE' : 'outputs/models/xgb_smote.pkl',
        }
        colors_list = [COLORS['neutre'], COLORS['ctgan'], COLORS['tvae'], COLORS['sinistre']]

        if df_pre is not None:
            from sklearn.model_selection import train_test_split
            X = df_pre.drop(columns=['claim_status'])
            y = df_pre['claim_status']
            _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

            fig, ax = plt.subplots(figsize=(8,6))
            for (name, path), color in zip(model_files.items(), colors_list):
                model = load_model(path)
                if model:
                    y_prob = model.predict_proba(X_test)[:,1]
                    auc    = roc_auc_score(y_test, y_prob)
                    fpr, tpr, _ = roc_curve(y_test, y_prob)
                    ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})", color=color, linewidth=2)
            ax.plot([0,1],[0,1],'k--',linewidth=1,label='Hasard (0.5)')
            ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
            ax.set_title("Courbe ROC", fontweight='bold'); ax.legend(fontsize=9)
            plt.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown('<div class="info-box">📌 Le Recall (52.4% pour le baseline) est la métrique prioritaire en assurance — manquer un sinistre est plus coûteux qu\'un faux positif.</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="section-header">Qualité des données synthétiques</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Score de Silhouette**")
            st.markdown('<div class="danger-box">⚠️ CTGAN : 0.7437 | TVAE : 0.7544<br>Insuffisant — problème d\'alignement d\'échelle</div>', unsafe_allow_html=True)
            st.markdown("**Perplexité**")
            st.markdown('<div class="danger-box">⚠️ Accuracy discriminateur = 1.0<br>Synthétiques trop différents des réels</div>', unsafe_allow_html=True)
        with col2:
            st.markdown("**TSTR**")
            st.markdown('<div class="warning-box">⚠️ CTGAN AUC=0.5156 | TVAE AUC=0.5540<br>Quasi-aléatoire seuls</div>', unsafe_allow_html=True)
            st.markdown("**Explication**")
            st.markdown('<div class="info-box">📌 Générateur entraîné sur données naturelles, classifieur sur données normalisées → décalage d\'échelle détecté.</div>', unsafe_allow_html=True)

        if df_ks is not None:
            st.markdown("---")
            st.dataframe(df_ks.round(4), use_container_width=True)

    with tab3:
        sens_choice = st.selectbox("Axe de sensibilité", [
            "Seuil de décision",
            "Volume de synthétiques",
            "Hyperparamètres",
            "Modèle réduit vs complet"
        ])

        if sens_choice == "Seuil de décision":
            st.markdown('<div class="success-box">✅ Seuil optimal : 0.50 — F1=0.1726, Recall=0.5240</div>', unsafe_allow_html=True)
            img = 'outputs/figures/12_sensibilite_seuil.png'
            if os.path.exists(img): st.image(img, use_column_width=True)

        elif sens_choice == "Volume de synthétiques":
            if df_vol is not None:
                st.dataframe(df_vol.round(4), use_container_width=True)
                fig, axes = plt.subplots(1,2, figsize=(12,5))
                axes[0].plot(df_vol['n_synthetic'], df_vol['auc'], 'o-', color=COLORS['neutre'], linewidth=2.5, markersize=8)
                axes[0].axhline(y=0.6454, color='gray', linestyle='--', linewidth=1.5, label='Baseline')
                axes[0].set_xlabel("N synthétiques"); axes[0].set_ylabel("AUC-ROC"); axes[0].legend()
                axes[1].plot(df_vol['n_synthetic'], df_vol['f1'], 'o-', color=COLORS['sinistre'], linewidth=2.5, markersize=8)
                axes[1].axhline(y=0.1726, color='gray', linestyle='--', linewidth=1.5, label='Baseline')
                axes[1].set_xlabel("N synthétiques"); axes[1].set_ylabel("F1-score"); axes[1].legend()
                plt.tight_layout(); st.pyplot(fig); plt.close()
                st.markdown('<div class="success-box">✅ Volume optimal : N=1 000</div>', unsafe_allow_html=True)

        elif sens_choice == "Hyperparamètres":
            img = 'outputs/figures/14_sensibilite_hyperparametres.png'
            if os.path.exists(img): st.image(img, use_column_width=True)
            st.markdown('<div class="success-box">✅ Meilleure config : depth=3, lr=0.05 → AUC=0.6662</div>', unsafe_allow_html=True)

        elif sens_choice == "Modèle réduit vs complet":
            comp = pd.DataFrame({
                'Modèle': ['Complet (93 vars)', 'Réduit (5 vars)', 'Différence'],
                'AUC': [0.6454, 0.6292, -0.0162],
                'F1': [0.1726, 0.1576, -0.0150],
                'Recall': [0.5240, 0.4947, -0.0293]
            })
            st.dataframe(comp, use_container_width=True)
            st.markdown('<div class="warning-box">⚠️ Modèle réduit perd -1.62% AUC mais gagne en stabilité</div>', unsafe_allow_html=True)

elif page == "⚠️ Limites et défis":
    st.markdown('<div class="main-title">⚠️ Limites Techniques et Défis du Projet</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Analyse critique des constatations et limitations observées</div>', unsafe_allow_html=True)
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Données", "🤖 Modèles", "⚙️ Infrastructure", "🔒 Sécurité"])

    with tab1:
        st.markdown('<div class="section-header">1️⃣ Qualité et Biais des Données</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="danger-box"><strong>Déséquilibre de classes extrême</strong></div>', unsafe_allow_html=True)
            st.markdown("""
            - **Ratio 1:14** (1 sinistre pour 14 non-sinistres)
            - Seulement **6.4%** de sinistres (3 746 sur 58 592)
            - **Impacts:** 
              - Biais du modèle vers la majorité (non-sinistres)
              - Metrics de classification inadaptées (accuracy 93% = inutile)
              - Recall faible malgré tout (52.4% baseline)
            """)
            
            st.markdown('<div class="warning-box"><strong>Données incomplètes/manquantes</strong></div>', unsafe_allow_html=True)
            st.markdown("""
            - **Valeurs NULL détectées** lors du prétraitement
            - **Outliers conservés** = peut biaiser les générateurs
            - **Typage imprécis** : colonnes texte convertibles en numérique
            """)
        
        with col2:
            st.markdown('<div class="danger-box"><strong>Contexte métier manquant</strong></div>', unsafe_allow_html=True)
            st.markdown("""
            - Les données brutes **n'expliquent pas les sinistres** (corrélations faibles)
            - Variables manquantes critiques:
              - Historique de sinistres du client
              - Infractions au code de la route
              - Région géographique exacte
              - Conditions météo/saisonnalité
            - **Conséquence:** Générateurs entraînés sur signaux faibles
            """)
            
            st.markdown('<div class="info-box"><strong>Passage à l\'échelle des données</strong></div>', unsafe_allow_html=True)
            st.markdown("""
            - Dataset de **58K polices** = modeste pour Deep Learning
            - Real-world: **millions de polices** requis
            - **Coût de stockage:** 13 GB (brut) → 1 GB (encodé)
            """)

    with tab2:
        st.markdown('<div class="section-header">2️⃣ Surapprentissage et Performance des Modèles</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="danger-box"><strong>Synthétiques de mauvaise qualité</strong></div>', unsafe_allow_html=True)
            st.markdown("""
            **Score de Silhouette:**
            - CTGAN: 0.7437 | TVAE: 0.7544
            - **Seuil acceptable: > 0.5** (ok mais limite)
            - **Problème détecté:** écarts d'échelle entre réelles et synthétiques
            
            **Discriminateur 100% accurate:**
            - Le classifieur distingue parfaitement réel vs synthétique
            - = synthétiques **trop éloignés** de la distribution réelle
            - = peu utiles pour augmentation robuste
            """)
            
            st.markdown('<div class="warning-box"><strong>Test Train-Synthetic-Real (TSTR)</strong></div>', unsafe_allow_html=True)
            st.markdown("""
            - CTGAN seul: AUC = 0.5156 (quasi-aléatoire)
            - TVAE seul: AUC = 0.5540 (quasi-aléatoire)
            - **= synthétiques NE REPRÉSENTENT PAS les vrais sinistres**
            """)
        
        with col2:
            st.markdown('<div class="warning-box"><strong>Surapprentissage XGBoost</strong></div>', unsafe_allow_html=True)
            st.markdown("""
            **Observations:**
            - Baseline AUC: 0.6454 (bon)
            - + CTGAN AUC: 0.5990 (**pire!**)
            - + TVAE AUC: 0.5956 (**pire!**)
            
            **Causes probables:**
            1. Synthétiques trop divergents (confondent le classifieur)
            2. Sur-régularisation du classifieur
            3. Dérive de distribution non corrigée
            
            **Impact métier:** Augmentation de données nuisible
            """)
            
            st.markdown('<div class="danger-box"><strong>Généralisation fragile</strong></div>', unsafe_allow_html=True)
            st.markdown("""
            - **Test set: 11K polices** = petit
            - Modèle réduit (5 vars vs 93) perd -1.6% AUC
            - = Performance instable aux changements de données
            - Pas de test sur **données futures** (temporal validation)
            """)

    with tab3:
        st.markdown('<div class="section-header">3️⃣ Coûts, Latence et Passage à l\'Échelle</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="warning-box"><strong>Latence de génération LLM</strong></div>', unsafe_allow_html=True)
            st.markdown("""
            **Temps d'exécution observé:**
            - Ollama/phi3.5: ~2-3 secondes par description
            - 10 descriptions: ~25 secondes
            - 1000 descriptions: **~40 minutes**
            - 1 million descriptions: **28 jours** ⚠️
            
            **Problème:** Inacceptable pour production real-time
            
            **Solutions partielles:**
            - Batch processing (déjà implémenté)
            - Modèle réduit + GPU
            - API cloud (Claude/GPT) mais coûteux
            """)
            
            st.markdown('<div class="warning-box"><strong>Stockage des modèles</strong></div>', unsafe_allow_html=True)
            st.markdown("""
            **Empreinte disque:**
            - CTGAN model: 5.3 MB
            - TVAE model: 2.1 MB
            - XGBoost models (4x): ~3.8 MB
            - **Total: ~12 MB** (acceptable)
            
            **Mais à l'échelle:**
            - Retraînement quotidien → versioning complexe
            - Multimodèles (par région/client) → explosion
            """)
        
        with col2:
            st.markdown('<div class="danger-box"><strong>Coûts de calcul</strong></div>', unsafe_allow_html=True)
            st.markdown("""
            **Entraînement actuel (CPU local):**
            - CTGAN 300 epochs: ~30 minutes
            - TVAE 300 epochs: ~30 minutes
            - XGBoost 4 modèles: ~5 minutes
            - **Total par run: ~65 minutes**
            
            **À l'échelle quotidienne:**
            - 1 run/jour: 65 min × 365 = 24,725 heures/an
            - GPU (AWS): ~$0.50/h → **$12K+/an**
            - Scaling: millions de polices → $100K+/an
            
            **ROI questionnable** pour assureurs petits/moyens
            """)
            
            st.markdown('<div class="warning-box"><strong>Absence de cache</strong></div>', unsafe_allow_html=True)
            st.markdown("""
            - Chaque appel LLM génère une réponse new
            - Pas de mémoire de descriptions passées
            - = Recalcul inutile + variation textuelle
            """)

    with tab4:
        st.markdown('<div class="section-header">4️⃣ Sécurité, Confidentialité et Conformité</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="danger-box"><strong>Données brutes sur disque</strong></div>', unsafe_allow_html=True)
            st.markdown("""
            **Risques détectés:**
            - CSV non chiffrés dans `outputs/`
            - Noms de polices encodés mais décodables
            - Données synthétiques = nouvelles données (à protéger)
            
            **Conformité RGPD:**
            - ❌ Pas d'anonymisation confirmée
            - ❌ Droit à l'oubli non implémenté
            - ❌ Pas d'audit trail des accès
            
            **Recommandations:**
            - Chiffrer au repos (AES-256)
            - RBAC sur fichiers
            - Logs d'accès centralisés
            """)
            
            st.markdown('<div class="warning-box"><strong>API LLM insécurisée</strong></div>', unsafe_allow_html=True)
            st.markdown("""
            **État actuel:**
            - Ollama en local HTTP (port 11434)
            - Pas d'authentification
            - Exposé au réseau local uniquement
            
            **Risques si déployé:**
            - Man-in-the-middle sur réseau corps
            - Injection de prompts (prompt injection attacks)
            - Pas de rate limiting → DDoS vulnérable
            """)
        
        with col2:
            st.markdown('<div class="danger-box"><strong>Biais discriminatoire</strong></div>', unsafe_allow_html=True)
            st.markdown("""
            **Dépistés (potentiels):**
            - Modèle entraîné sur données imbalancées
            - = **surprédiction sinistres chez groupes minoritaires**
            - (ex: certaines régions, tranches d'âge)
            
            **Impact légal:**
            - ❌ Discrimination algorithmique (illégale)
            - ❌ Pénalités: CNIL, tribunaux
            
            **Test de fairness manquant:**
            - Pas de métriques par groupe démographique
            - Pas de calibration post-hoc
            
            **À faire:**
            - Fairness audit complet (eg: Fairness Indicators)
            - Débiaiser modèles si nécessaire
            """)
            
            st.markdown('<div class="info-box"><strong>Données synthétiques = RE-identification possible</strong></div>', unsafe_allow_html=True)
            st.markdown("""
            **Danger peu connu:**
            - GAN-generated data ≠ anonyme garantie
            - Études montrent: possible re-identifier individus
            - Synthétiques CTGAN/TVAE pas plus sûrs
            
            **Solution:** Combinaison
            - Differential privacy en entraînement
            - Audits de re-identification
            """)

    # SYNTHÈSE FINALE
    st.markdown("---")
    st.markdown('<div class="section-header">📋 Résumé des Limitations Critiques</div>', unsafe_allow_html=True)
    
    limitations = pd.DataFrame({
        'Catégorie': [
            'Données', 'Données', 'Données',
            'Modèles', 'Modèles', 'Modèles',
            'Infrastructure', 'Infrastructure',
            'Sécurité', 'Sécurité'
        ],
        'Limitation': [
            'Déséquilibre 1:14', 'Contexte métier faible', 'Taille modeste',
            'Synthétiques mauvaise qualité (TSTR AUC~0.55)', 'Augmentation nuisible', 'Généralisation fragile',
            'Latence LLM excessive (28j/1M)', 'Coûts de calcul élevés',
            'Pas de chiffrement données', 'Biais discriminatoire non audité'
        ],
        'Sévérité': ['Critique', 'Moyen', 'Moyen',
                     'Critique', 'Critique', 'Moyen',
                     'Moyen', 'Moyen',
                     'Critique', 'Critique'],
        'Mitigation': [
            'SMOTE local, class weights', 'Ajouter features métier',  'Augmenter data collection',
            'Vérifier architecture GAN', 'Tester CTGAN v1.7+', 'Cross-validation temporelle',
            'GPU + batching', 'Cached + approximate',
            'AES-256 + RBAC', 'Fairness audit + differential privacy'
        ]
    })
    
    st.dataframe(limitations, use_container_width=True)
    
    st.markdown('---')
    st.markdown('<div class="section-header">🚀 Recommandations Prioritaires (Roadmap)</div>', unsafe_allow_html=True)
    st.markdown("""
    ### Phase 1 (Court terme - 1-2 mois)
    1. **Fairness Audit** — Vérifier absence de biais par démographique
    2. **Chiffrement données** — Protéger CSVs sur disque
    3. **TSTR investigation** — Diagnostiquer pourquoi synthétiques échouent seuls
    4. **Threshold tuning** — Optimiser seuil décision (pas 0.5 fixe)
    
    ### Phase 2 (Moyen terme - 3-6 mois)
    1. **Ajouter features métier** — Historique sinistres, géolocalisation fine
    2. **Réévaluer CTGAN/TVAE** — Tenter normalization fix ou hyperparams
    3. **GPU infrastructure** — Réduire latence LLM de 28j → 2-3 jours
    4. **Differential privacy** — Synthétiques provably private
    
    ### Phase 3 (Long terme - 6-12 mois)
    1. **Modèle Ensemble** — Combiner CTGAN + TVAE + SMOTE intelligemment
    2. **Temporal validation** — Test modèle sur futures données
    3. **RBAC + audit logs** — Conformité RGPD complète
    4. **Production pipeline** — Retraining daily automatisé
    """)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — PRÉDICTION INDIVIDUELLE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Prédiction individuelle":
    st.markdown('<div class="main-title">🔮 Prédiction du risque sinistre</div>', unsafe_allow_html=True)

    xgb_baseline = load_model('outputs/models/xgb_baseline.pkl')
    if xgb_baseline is None or df_pre is None:
        st.error("Modèle non disponible."); st.stop()

    st.markdown('<div class="success-box">✅ Modèle XGBoost Baseline chargé</div>', unsafe_allow_html=True)
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        customer_age        = st.slider("Âge du client", 35, 75, 45)
        vehicle_age         = st.slider("Âge du véhicule (années)", 0, 20, 3)
        subscription_length = st.slider("Durée de la police (années)", 0, 14, 5)
    with col2:
        region_density = st.number_input("Densité de la région", 290, 73430, 8000)
        airbags        = st.slider("Nombre d'airbags", 1, 6, 2)
        ncap_rating    = st.slider("Note NCAP", 0, 5, 3)
    with col3:
        cylinder     = st.selectbox("Cylindres", [3, 4], index=1)
        displacement = st.number_input("Cylindrée (cm³)", 796, 1498, 1197)
        gross_weight = st.number_input("Poids total (kg)", 1051, 1720, 1335)

    if st.button("🔮 Prédire le risque sinistre", use_container_width=True):
        feature_cols  = df_pre.drop(columns=['claim_status']).columns
        X_input       = pd.DataFrame(0, index=[0], columns=feature_cols)
        numeric_means = df_pre.drop(columns=['claim_status']).mean()
        numeric_stds  = df_pre.drop(columns=['claim_status']).std()

        def norm(col, val):
            if col in numeric_stds.index and numeric_stds[col] > 0:
                return (val - numeric_means[col]) / numeric_stds[col]
            return val

        for col, val in [('customer_age',customer_age),('vehicle_age',vehicle_age),
                          ('subscription_length',subscription_length),('region_density',region_density),
                          ('airbags',airbags),('ncap_rating',ncap_rating),
                          ('cylinder',cylinder),('displacement',displacement),('gross_weight',gross_weight)]:
            if col in X_input.columns: X_input[col] = norm(col, val)

        prob = xgb_baseline.predict_proba(X_input)[0][1]

        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Probabilité de sinistre", f"{prob*100:.1f}%")
        with col2: st.metric("Prédiction", "🚨 Sinistre" if prob > 0.5 else "✅ Pas de sinistre")
        with col3: st.metric("Taux moyen dataset", "6.4%")

        fig, ax = plt.subplots(figsize=(8,2))
        ax.barh([0],[prob], color=COLORS['sinistre'] if prob>0.3 else COLORS['accent'], height=0.4)
        ax.barh([0],[1-prob], left=[prob], color='#EEEEEE', height=0.4)
        ax.axvline(x=0.064, color='gray', linestyle='--', linewidth=1.5, label='Moy. (6.4%)')
        ax.set_xlim(0,1); ax.set_yticks([])
        ax.set_xlabel("Probabilité"); ax.set_title(f"Risque : {prob*100:.1f}%", fontweight='bold')
        ax.legend(); plt.tight_layout(); st.pyplot(fig); plt.close()

        if prob > 0.3:
            st.markdown('<div class="danger-box">⚠️ <strong>Profil à risque élevé</strong></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="success-box">✅ <strong>Profil standard</strong></div>', unsafe_allow_html=True)

        # Description LLM automatique
        st.markdown("---")
        st.markdown('<div class="section-header">🧠 Description LLM de ce profil</div>', unsafe_allow_html=True)
        try:
            from openai import OpenAI
            client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
            prompt = f"""Tu es un expert en sinistres d'assurance automobile. Rédige une description professionnelle de 50-80 mots de ce profil:
- Âge client: {customer_age} ans
- Âge véhicule: {vehicle_age} ans
- Durée police: {subscription_length} ans
- Densité région: {region_density}
- Airbags: {airbags}
- Note NCAP: {ncap_rating}/5
- Probabilité sinistre estimée: {prob*100:.1f}%"""
            response = client.chat.completions.create(
                model="phi3.5",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300, temperature=0.2
            )
            description = response.choices[0].message.content.strip()
            st.markdown(f'<div class="llm-box">📝 {description}</div>', unsafe_allow_html=True)
        except:
            st.markdown('<div class="warning-box">⚠️ Ollama non disponible pour la description LLM</div>', unsafe_allow_html=True)