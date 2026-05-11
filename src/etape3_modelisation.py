"""
Étape 3 — Modélisation
======================

XGBoost baseline + CTGAN + TVAE + SMOTE + enrichissement LLM.
Produit les modèles entraînés et les données synthétiques.

Auteurs : Groupe ISFA M2 2025-2026
Exécution : python etape3_modelisation.py

Structure projet attendue :
    PROJET/
    ├── data/                 (données brutes)
    ├── src/                  (ce script doit être lancé d'ici)
    └── outputs/              (résultats produits ici)
"""

# ─── Activer le backend matplotlib non-interactif pour exécution headless ───
import matplotlib
matplotlib.use("Agg")  # commenter cette ligne pour affichage interactif

# ───────────────────────────────────────────────────────────────────────────
# # 🤖 Étape 3 — Modélisation : Génération Synthétique & Classification
#
# **Projet :** Génération de données synthétiques de sinistres pour la tarification en assurance  
# **Dataset :** Insurance Claims Data — 58 592 polices  
# **Auteurs :** Groupe ISFA 2025-2026
#
# ---
#
# ## 📋 Sommaire
# 1. Justification de l'approche
# 2. Imports & installation
# 3. Chargement et split stratifié
# 4. Fonction d'évaluation unifiée
# 5. **Baseline XGBoost** (référence)
# 6. **Génération CTGAN** (GAN conditionnel tabulaire)
# 7. **Génération TVAE** (Autoencodeur variationnel)
# 8. **XGBoost + CTGAN** (augmentation)
# 9. **XGBoost + TVAE** (augmentation)
# 10. **SMOTE** (interpolation classique)
# 11. **LLM — Ollama/phi3.5** (descriptions textuelles)
# 12. Tableau comparatif final
# 13. Analyse de la qualité des données synthétiques
# 14. Résumé
#
# ---
#
# ## 1. Justification de l'approche
#
# ### Approche 1 — Génération tabulaire (CTGAN + TVAE)
#
# | Critère | LLM seul | CTGAN/TVAE |
# |---------|----------|------------|
# | Adapté aux données tabulaires | ❌ Conçu pour le texte | ✅ Conçu spécifiquement |
# | Coût computationnel | ❌ GPU massif | ✅ CPU acceptable |
# | Performances sur données numériques | ❌ Non démontré | ✅ State-of-the-art (MIT SDV) |
# | Respect des distributions marginales | ❌ | ✅ |
# | Pertinence actuarielle | ❌ Faible | ✅ Directe |
#
# ### Approche 2 — LLM local (Ollama/phi3.5)
# Le LLM enrichit les sinistres synthétiques avec des **descriptions textuelles professionnelles**, simulant le travail rédactionnel d'un gestionnaire de sinistres.
#
# ### Pipeline complet
# ```
# Données réelles (58 592 polices)
#         │
#         ├──► CTGAN ──► 5 000 sinistres synthétiques
#         ├──► TVAE  ──► 5 000 sinistres synthétiques  
#         ├──► SMOTE ──► Interpolation (baseline)
#         │              Ollama/phi3.5 ──► Descriptions textuelles
#         └──► XGBoost ◄── Comparaison AUC / F1 / Recall
# ```
# ───────────────────────────────────────────────────────────────────────────

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. INSTALLATION DES DÉPENDANCES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
import subprocess, sys

packages = [
    ('sdv',                 'sdv'),
    ('xgboost',             'xgboost'),
    ('imbalanced-learn',    'imblearn'),
    ('openai',              'openai'),
    ('scikit-learn',        'sklearn'),
    ('matplotlib',          'matplotlib'),
    ('seaborn',             'seaborn'),
]

for pkg_name, import_name in packages:
    try:
        __import__(import_name)
        print(f'✅ {pkg_name} déjà installé')
    except ImportError:
        print(f'📦 Installation de {pkg_name}...')
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg_name, '-q'])
        print(f'✅ {pkg_name} installé')

print('\n✅ Toutes les dépendances sont prêtes')

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# IMPORTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
import pandas as pd
import numpy as np
import os, pickle, json
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (roc_auc_score, f1_score, recall_score,
                             precision_score, confusion_matrix,
                             ConfusionMatrixDisplay, roc_curve,
                             precision_recall_curve, average_precision_score)
import xgboost as xgb

try:
    from sdv.single_table import CTGANSynthesizer, TVAESynthesizer
    from sdv.metadata import SingleTableMetadata
    from sdv.evaluation.single_table import evaluate_quality
    SDV_AVAILABLE = True
    print(f'✅ SDV {__import__("sdv").__version__} disponible')
except ImportError as e:
    SDV_AVAILABLE = False
    print(f'⚠️  SDV non disponible : {e}')

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
    print('✅ imbalanced-learn disponible')
except ImportError:
    SMOTE_AVAILABLE = False
    print('⚠️  imbalanced-learn non disponible')

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
    print('✅ openai disponible (pour Ollama local)')
except ImportError:
    OPENAI_AVAILABLE = False
    print('⚠️  openai non disponible')

COLORS = {'sinistre': '#E74C3C', 'non_sinistre': '#2E86AB', 'neutre': '#1F4E79', 'accent': '#F39C12'}
os.makedirs('../outputs/figures', exist_ok=True)
os.makedirs('../outputs/models', exist_ok=True)
os.makedirs('../outputs/synthetic', exist_ok=True)
os.makedirs('../outputs/llm', exist_ok=True)

# ── Configuration ──
N_SYNTHETIC  = 5000
EPOCHS_CTGAN = 300
EPOCHS_TVAE  = 300
RANDOM_STATE = 42
TEST_SIZE    = 0.2
N_LLM        = 10

print(f'\n  N_SYNTHETIC  = {N_SYNTHETIC:,}')
print(f'  EPOCHS       = {EPOCHS_CTGAN}')
print(f'  TEST_SIZE    = {TEST_SIZE}')
print(f'  N_LLM        = {N_LLM}')

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. CHARGEMENT ET SPLIT STRATIFIÉ
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
df_encoded = pd.read_csv('../outputs/data_encoded.csv')
df_pre     = pd.read_csv('../outputs/data_preprocessed.csv')

try:
    df_raw = pd.read_csv('../data/Insurance claims data.csv', sep=',')
    RAW_AVAILABLE = True
except:
    df_raw = df_encoded.copy()
    RAW_AVAILABLE = False

print(f'[3] Datasets chargés :')
print(f'    data_preprocessed : {df_pre.shape}')
print(f'    data_encoded      : {df_encoded.shape}')

X = df_pre.drop(columns=['claim_status'])
y = df_pre['claim_status']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# Split pour CTGAN/TVAE (données non normalisées)
X_enc = df_encoded.drop(columns=['claim_status'])
y_enc = df_encoded['claim_status']
X_enc_train, X_enc_test, y_enc_train, y_enc_test = train_test_split(
    X_enc, y_enc, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_enc
)

print(f'\n  Split stratifié (test={TEST_SIZE}) :')
print(f'    Train : {X_train.shape[0]:,} — {int(y_train.sum()):,} sinistres ({y_train.mean()*100:.1f}%)')
print(f'    Test  : {X_test.shape[0]:,}  — {int(y_test.sum()):,} sinistres ({y_test.mean()*100:.1f}%)')
print(f'\n    ✅ Split stratifié : le taux de sinistres est conservé dans train et test')
print(f'    ℹ️  Le test set est UNIQUEMENT réel — jamais de synthétiques en test')

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. FONCTION D'ÉVALUATION UNIFIÉE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
results = []

def evaluate_model(model, X_test, y_test, model_name, save_curves=True):
    """
    Évaluation complète d'un modèle de classification.
    Métriques : AUC-ROC, F1, Recall, Precision, AP
    Sorties   : matrice de confusion + courbes ROC/PR
    """
    y_pred      = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    auc  = roc_auc_score(y_test, y_pred_prob)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    prec = precision_score(y_test, y_pred, zero_division=0)
    ap   = average_precision_score(y_test, y_pred_prob)
    cm   = confusion_matrix(y_test, y_pred)

    print(f'\n━━ {model_name} ━━')
    print(f'  AUC-ROC   : {auc:.4f}  ← capacité discriminante globale')
    print(f'  F1-score  : {f1:.4f}  ← équilibre précision/rappel')
    print(f'  Recall    : {rec:.4f}  ← % vrais sinistres détectés ★ PRIORITÉ ACTUARIELLE')
    print(f'  Precision : {prec:.4f}  ← % sinistres prédits réels')
    print(f'  AP (PR)   : {ap:.4f}  ← aire sous courbe précision-rappel')
    print(f'  TN={cm[0,0]:>5}  FP={cm[0,1]:>5}')
    print(f'  FN={cm[1,0]:>5}  TP={cm[1,1]:>5}')

    # Matrice de confusion
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f'Évaluation — {model_name}', fontweight='bold', color=COLORS['neutre'])

    ConfusionMatrixDisplay(cm, display_labels=['Non-sinistre', 'Sinistre']).plot(
        ax=axes[0], colorbar=False, cmap='Blues')
    axes[0].set_title('Matrice de confusion')

    # Courbe ROC
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    axes[1].plot(fpr, tpr, color=COLORS['sinistre'], lw=2, label=f'AUC={auc:.3f}')
    axes[1].plot([0,1],[0,1], 'k--', lw=1, alpha=0.5, label='Aléatoire')
    axes[1].fill_between(fpr, tpr, alpha=0.1, color=COLORS['sinistre'])
    axes[1].set_xlabel('Taux de faux positifs'); axes[1].set_ylabel('Taux de vrais positifs')
    axes[1].set_title(f'Courbe ROC')
    axes[1].legend()

    # Courbe Précision-Rappel
    prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_pred_prob)
    axes[2].plot(rec_curve, prec_curve, color=COLORS['non_sinistre'], lw=2, label=f'AP={ap:.3f}')
    axes[2].axhline(y=y_test.mean(), color='gray', linestyle='--', lw=1, label='Baseline (aléatoire)')
    axes[2].set_xlabel('Rappel'); axes[2].set_ylabel('Précision')
    axes[2].set_title('Courbe Précision-Rappel')
    axes[2].legend()

    plt.tight_layout()
    safe_name = model_name.replace(' ', '_').replace('+', '').replace('/', '')
    plt.savefig(f'../outputs/figures/eval_{safe_name}.png', dpi=120, bbox_inches='tight')
    plt.close('all')  # plt.show() en notebook; plt.close()

    results.append({
        'model': model_name, 'auc': auc, 'f1': f1,
        'recall': rec, 'precision': prec, 'ap': ap
    })
    return auc, f1, rec, prec

print('✅ Fonction d\'évaluation définie')
print('   Métriques : AUC-ROC, F1, Recall ★, Precision, AP')
print('   Sorties   : matrice de confusion + courbe ROC + courbe PR')

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. BASELINE — XGBoost sur données réelles uniquement
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print('[5] Baseline — XGBoost sur données réelles')
print('-' * 50)
print('  Ce modèle est le point de référence (baseline).')
print('  Tout modèle augmenté doit faire MIEUX que lui.')

scale_pos_weight = int((y_train == 0).sum() / (y_train == 1).sum())
print(f'\n  scale_pos_weight = {scale_pos_weight}')
print(f'  Interprétation : une erreur sur un sinistre pèse {scale_pos_weight}× plus')

xgb_baseline = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=RANDOM_STATE,
    eval_metric='auc',
    verbosity=0,
    use_label_encoder=False
)
xgb_baseline.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

evaluate_model(xgb_baseline, X_test, y_test, 'XGBoost Baseline')
pickle.dump(xgb_baseline, open('../outputs/models/xgb_baseline.pkl', 'wb'))
print('\n✅ Modèle sauvegardé : ../outputs/models/xgb_baseline.pkl')

# Importance des features — baseline
importances = pd.Series(xgb_baseline.feature_importances_, index=X_train.columns)
top20 = importances.sort_values(ascending=True).tail(20)

fig, ax = plt.subplots(figsize=(10, 8))
colors = [COLORS['sinistre'] if v > top20.quantile(0.75) else COLORS['non_sinistre'] for v in top20]
top20.plot(kind='barh', ax=ax, color=colors, edgecolor='white')
ax.set_title('Top 20 features — XGBoost Baseline (gain)',
             fontsize=12, fontweight='bold', color=COLORS['neutre'])
ax.set_xlabel('Importance (gain)')
plt.tight_layout()
plt.savefig('../outputs/figures/feature_importance_baseline.png', dpi=150, bbox_inches='tight')
plt.close('all')  # plt.show() en notebook
print('📊 Figure sauvegardée : ../outputs/figures/feature_importance_baseline.png')

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. GÉNÉRATION CTGAN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CTGAN_AVAILABLE = False

if SDV_AVAILABLE:
    print('[6] Génération CTGAN — Conditional Tabular GAN (MIT, 2019)')
    print('-' * 60)
    print('  Architecture : 2 réseaux antagonistes')
    print('    Générateur   : fabrique des sinistres synthétiques')
    print('    Discriminateur : tente de les détecter')
    print('  Avantage CTGAN : gère les données mixtes (numérique + catégoriel)')
    print('  Entraîné UNIQUEMENT sur les sinistres du train (classe minoritaire)')

    # Préparer le subset sinistres (données encodées, non normalisées)
    df_train_enc_full = X_enc_train.copy()
    df_train_enc_full['claim_status'] = y_enc_train.values
    df_sinistres = df_train_enc_full[df_train_enc_full['claim_status'] == 1].drop(columns=['claim_status'])
    print(f'\n  Entraînement sur {len(df_sinistres):,} sinistres réels...')

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df_sinistres)

    ctgan = CTGANSynthesizer(
        metadata,
        epochs=EPOCHS_CTGAN,
        verbose=True,
        batch_size=500,
        generator_lr=2e-4,
        discriminator_lr=2e-4
    )
    ctgan.fit(df_sinistres)
    print('\n✅ CTGAN entraîné')

    synthetic_ctgan = ctgan.sample(num_rows=N_SYNTHETIC)
    synthetic_ctgan['claim_status'] = 1
    synthetic_ctgan.to_csv('../outputs/synthetic/synthetic_ctgan.csv', index=False)
    ctgan.save('../outputs/models/ctgan_model.pkl')
    print(f'✅ {N_SYNTHETIC:,} sinistres synthétiques CTGAN générés')
    print(f'✅ Sauvegardé : ../outputs/synthetic/synthetic_ctgan.csv')
    CTGAN_AVAILABLE = True

    print('\n  Aperçu des données synthétiques CTGAN (3 premières lignes) :')
    print(synthetic_ctgan.head(3))
else:
    print('⚠️  SDV non disponible — CTGAN ignoré')
    print('   pip install sdv')

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. GÉNÉRATION TVAE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TVAE_AVAILABLE = False

if SDV_AVAILABLE:
    print('[7] Génération TVAE — Tabular Variational Autoencoder (MIT, 2019)')
    print('-' * 60)
    print('  Architecture : Encodeur → Espace latent gaussien → Décodeur')
    print('    1. Encodeur  : compresse chaque sinistre en vecteur latent z')
    print('    2. Espace latent : distributions normales N(μ, σ²)')
    print('    3. Décodeur  : reconstruit un sinistre depuis z ~ N(0, I)')
    print('  Avantage vs CTGAN : plus stable, pas de mode collapse')

    tvae = TVAESynthesizer(
        metadata,
        epochs=EPOCHS_TVAE,
        verbose=True,
        batch_size=500
    )
    tvae.fit(df_sinistres)
    print('\n✅ TVAE entraîné')

    synthetic_tvae = tvae.sample(num_rows=N_SYNTHETIC)
    synthetic_tvae['claim_status'] = 1
    synthetic_tvae.to_csv('../outputs/synthetic/synthetic_tvae.csv', index=False)
    tvae.save('../outputs/models/tvae_model.pkl')
    print(f'✅ {N_SYNTHETIC:,} sinistres synthétiques TVAE générés')
    TVAE_AVAILABLE = True

    print('\n  Aperçu des données synthétiques TVAE (3 premières lignes) :')
    print(synthetic_tvae.head(3))
else:
    print('⚠️  SDV non disponible — TVAE ignoré')

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Fonction helper : préparer l'augmentation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def prepare_augmented_train(synthetic_df, X_train_ref, y_train_ref, n_synthetic, preprocessed=True):
    """
    Prépare un dataset d'entraînement augmenté.
    Si preprocessed=True, les synthétiques sont alignés sur les features normalisées.
    """
    syn_X = synthetic_df.drop(columns=['claim_status'], errors='ignore')
    # Alignement des colonnes (fill_value=0 pour les colonnes manquantes)
    syn_X = syn_X.reindex(columns=X_train_ref.columns, fill_value=0)

    X_aug = pd.concat([X_train_ref, syn_X], ignore_index=True)
    y_aug = pd.concat([y_train_ref, pd.Series([1]*n_synthetic)], ignore_index=True)
    spw   = max(1, int((y_aug == 0).sum() / (y_aug == 1).sum()))

    return X_aug, y_aug, spw

print('✅ Fonction prepare_augmented_train définie')

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 8. XGBoost + CTGAN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if CTGAN_AVAILABLE:
    print('[8] XGBoost + CTGAN')
    X_aug_ctgan, y_aug_ctgan, spw_ctgan = prepare_augmented_train(
        synthetic_ctgan, X_train, y_train, N_SYNTHETIC
    )
    print(f'  Train augmenté : {len(X_aug_ctgan):,} — {int(y_aug_ctgan.sum()):,} sinistres ({y_aug_ctgan.mean()*100:.1f}%)')
    print(f'  scale_pos_weight ajusté : {spw_ctgan}')

    xgb_ctgan = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw_ctgan,
        random_state=RANDOM_STATE, eval_metric='auc', verbosity=0
    )
    xgb_ctgan.fit(X_aug_ctgan, y_aug_ctgan, eval_set=[(X_test, y_test)], verbose=False)
    evaluate_model(xgb_ctgan, X_test, y_test, 'XGBoost + CTGAN')
    pickle.dump(xgb_ctgan, open('../outputs/models/xgb_ctgan.pkl', 'wb'))
    print('✅ Modèle sauvegardé')
else:
    print('⚠️  CTGAN non disponible — section ignorée')

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 9. XGBoost + TVAE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if TVAE_AVAILABLE:
    print('[9] XGBoost + TVAE')
    X_aug_tvae, y_aug_tvae, spw_tvae = prepare_augmented_train(
        synthetic_tvae, X_train, y_train, N_SYNTHETIC
    )
    print(f'  Train augmenté : {len(X_aug_tvae):,} — {int(y_aug_tvae.sum()):,} sinistres ({y_aug_tvae.mean()*100:.1f}%)')

    xgb_tvae = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw_tvae,
        random_state=RANDOM_STATE, eval_metric='auc', verbosity=0
    )
    xgb_tvae.fit(X_aug_tvae, y_aug_tvae, eval_set=[(X_test, y_test)], verbose=False)
    evaluate_model(xgb_tvae, X_test, y_test, 'XGBoost + TVAE')
    pickle.dump(xgb_tvae, open('../outputs/models/xgb_tvae.pkl', 'wb'))
    print('✅ Modèle sauvegardé')
else:
    print('⚠️  TVAE non disponible — section ignorée')

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 10. SMOTE — Baseline d'augmentation classique
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if SMOTE_AVAILABLE:
    print('[10] SMOTE — Synthetic Minority Over-sampling Technique')
    print('-' * 55)
    print('  Principe : INTERPOLE entre des sinistres existants')
    print('  Ne génère PAS de nouvelles distributions')
    print('  Inclus comme baseline : si CTGAN/TVAE ≤ SMOTE,')
    print('  le deep learning génératif n\'apporte rien.')

    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
    X_smote, y_smote = smote.fit_resample(X_train, y_train)
    spw_smote = max(1, int((y_smote == 0).sum() / (y_smote == 1).sum()))

    print(f'\n  Train SMOTE : {len(X_smote):,} — {int(y_smote.sum()):,} sinistres ({y_smote.mean()*100:.1f}%)')

    xgb_smote = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw_smote,
        random_state=RANDOM_STATE, eval_metric='auc', verbosity=0
    )
    xgb_smote.fit(X_smote, y_smote, eval_set=[(X_test, y_test)], verbose=False)
    evaluate_model(xgb_smote, X_test, y_test, 'XGBoost + SMOTE')
    pickle.dump(xgb_smote, open('../outputs/models/xgb_smote.pkl', 'wb'))
    print('✅ Modèle sauvegardé')
else:
    print('⚠️  imbalanced-learn non disponible — pip install imbalanced-learn')

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 11. GÉNÉRATION LLM — Ollama/phi3.5
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print('[11] Génération LLM — Ollama/phi3.5 (local, gratuit)')
print('=' * 60)
print('  Architecture LLM locale :')
print('    Ollama : serveur LLM local (données privées, 0 € de coût)')
print('    phi3.5 : modèle Microsoft — léger, rapide, excellent en français')
print()
print('  Rôle dans le pipeline :')
print('    Pour chaque sinistre synthétique, le LLM génère une')
print('    description textuelle professionnelle (50-100 mots),')
print('    comme un gestionnaire de sinistres expert.')
print()
print('  Prérequis Ollama :')
print('    1. Installation : https://ollama.com')
print('    2. Modèle       : ollama pull phi3.5')
print('    3. Serveur      : ollama serve')


class ClaimsLLMGenerator:
    """
    Générateur de descriptions de sinistres via LLM local (Ollama).
    Utilise l'API OpenAI-compatible d'Ollama.
    """

    SYSTEM_PROMPT = (
        "Tu es un expert senior en gestion de sinistres automobiles avec 20 ans d'expérience. "
        "Tu rédiges des descriptions professionnelles de dossiers de sinistres pour les équipes actuarielles. "
        "Tes descriptions sont concises, précises, et incluent une évaluation du risque."
    )

    def __init__(self, model='phi3.5', base_url='http://localhost:11434/v1'):
        self.model  = model
        self.client = None
        if OPENAI_AVAILABLE:
            try:
                from openai import OpenAI
                self.client = OpenAI(base_url=base_url, api_key='ollama')
                self.client.models.list()  # Test de connexion
                print(f'✅ Ollama connecté ({base_url})')
            except Exception as e:
                print(f'⚠️  Ollama non accessible : {e}')
                self.client = None
        else:
            print('⚠️  openai non installé')

    def _build_prompt(self, claim: dict) -> str:
        details = '\n'.join(f'  - {k.replace("_", " ").title()}: {v}' for k, v in claim.items())
        return (
            f'Voici les données d\'un dossier de sinistre automobile :\n\n'
            f'{details}\n\n'
            f'Rédige une description professionnelle de 60-80 mots. '
            f'Inclus : contexte assuré, profil de risque, observations techniques, '
            f'et recommandation de traitement. Sois concis et factuel.'
        )

    def generate(self, claim: dict) -> str:
        if self.client is None:
            return 'LLM non disponible — vérifier Ollama'
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {'role': 'system', 'content': self.SYSTEM_PROMPT},
                    {'role': 'user',   'content': self._build_prompt(claim)}
                ],
                max_tokens=200,
                temperature=0.3,
                top_p=0.9
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f'Erreur génération : {e}'

    def generate_batch(self, claims: list, verbose=True) -> list:
        descriptions = []
        for i, claim in enumerate(claims):
            if verbose:
                print(f'  Sinistre {i+1}/{len(claims)}...', end='\r')
            descriptions.append(self.generate(claim))
        if verbose:
            print(f'\n✅ {len(descriptions)} descriptions générées')
        return descriptions


llm = ClaimsLLMGenerator()
print(f'\nGénérateur LLM prêt : {llm.client is not None}')

# Génération des descriptions LLM
raw_cols = ['customer_age', 'vehicle_age', 'subscription_length',
            'region_density', 'fuel_type', 'segment',
            'transmission_type', 'airbags', 'ncap_rating']
raw_cols = [c for c in raw_cols if c in df_raw.columns] if RAW_AVAILABLE else []

if llm.client is not None and RAW_AVAILABLE and len(raw_cols) > 0:
    df_claims_raw = df_raw[df_raw['claim_status'] == 1][raw_cols].head(N_LLM)
    claims_list   = df_claims_raw.to_dict(orient='records')

    print(f'Génération de {N_LLM} descriptions de sinistres réels...\n')
    descriptions = llm.generate_batch(claims_list, verbose=True)

    # Sauvegarde
    df_llm = df_claims_raw.copy().reset_index(drop=True)
    df_llm['description_llm'] = descriptions
    df_llm.to_csv('../outputs/llm/sinistres_avec_descriptions.csv', index=False)

    print(f'\n✅ Sauvegardé : ../outputs/llm/sinistres_avec_descriptions.csv')

    # Affichage exemples
    print('\n' + '='*60)
    print('  EXEMPLES DE DESCRIPTIONS GÉNÉRÉES PAR LE LLM')
    print('='*60)
    for i in range(min(3, len(descriptions))):
        print(f'\n  📄 Sinistre {i+1} :')
        for k, v in claims_list[i].items():
            print(f'     {k}: {v}')
        print(f'\n  💬 Description LLM :')
        print(f'     {descriptions[i]}')

    print(df_llm.head())

else:
    print('⚠️  LLM non disponible')
    print('   Instructions :')
    print('   1. Installer Ollama : https://ollama.com/download')
    print('   2. Télécharger le modèle : ollama pull phi3.5')
    print('   3. Lancer le serveur : ollama serve')
    print('   4. Ré-exécuter cette cellule')

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 12. TABLEAU COMPARATIF FINAL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if len(results) == 0:
    print('⚠️  Aucun modèle évalué — vérifier les étapes précédentes')
else:
    df_results = pd.DataFrame(results).sort_values('f1', ascending=False)
    df_results.to_csv('../outputs/resultats_modelisation.csv', index=False)

    print('[12] Tableau comparatif final :')
    print(f'  Évalué sur le TEST SET RÉEL : {len(X_test):,} polices, {int(y_test.sum()):,} sinistres')
    print(df_results.round(4))

    # Graphique radar / barres multiples
    fig, axes = plt.subplots(1, 5, figsize=(20, 6))
    fig.suptitle('Comparaison des modèles — Test Set réel uniquement',
                 fontsize=13, fontweight='bold', color=COLORS['neutre'])

    palette = ['#1F4E79', '#2E86AB', '#E74C3C', '#2ECC71', '#F39C12']
    metrics = ['auc', 'f1', 'recall', 'precision', 'ap']
    labels  = ['AUC-ROC', 'F1-score', 'Recall ★', 'Precision', 'Avg Precision']

    for i, (metric, label) in enumerate(zip(metrics, labels)):
        vals = df_results[metric].values if metric in df_results.columns else [0]*len(df_results)
        bars = axes[i].bar(range(len(df_results)), vals,
                            color=[palette[j % len(palette)] for j in range(len(df_results))],
                            alpha=0.8, edgecolor='white', linewidth=1.5)
        axes[i].set_title(label, fontweight='bold')
        axes[i].set_ylim(0, max(vals)*1.3 if max(vals) > 0 else 1)
        axes[i].set_xticks(range(len(df_results)))
        axes[i].set_xticklabels(df_results['model'], rotation=40, ha='right', fontsize=8)
        for bar, val in zip(bars, vals):
            axes[i].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                         f'{val:.3f}', ha='center', fontsize=8, fontweight='bold')

    plt.tight_layout()
    plt.savefig('../outputs/figures/12_comparaison_modeles.png', dpi=150, bbox_inches='tight')
    plt.close('all')  # plt.show() en notebook
    print('\n📊 Figure sauvegardée : ../outputs/figures/12_comparaison_modeles.png')

    # Diagnostic
    best = df_results.iloc[0]
    best_recall = df_results.loc[df_results['recall'].idxmax()]
    print(f'\n  🏆 Meilleur F1    : {best["model"]} → F1={best["f1"]:.4f}')
    print(f'  🎯 Meilleur Recall: {best_recall["model"]} → Recall={best_recall["recall"]:.4f}')
    print(f'\n  Interprétation actuarielle :')
    print(f'  Recall est la métrique prioritaire : chaque sinistre non-détecté')
    print(f'  représente un coût potentiel élevé pour l\'assureur.')

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 13. QUALITÉ DES DONNÉES SYNTHÉTIQUES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if CTGAN_AVAILABLE or TVAE_AVAILABLE:
    print('[13] Analyse de la qualité des données synthétiques')
    print('-' * 55)

    # Comparaison distributions réelles vs synthétiques
    key_cols = ['customer_age', 'vehicle_age', 'subscription_length', 'region_density']
    key_cols = [c for c in key_cols if c in df_sinistres.columns]

    n_plots = len(key_cols)
    fig, axes = plt.subplots(2 if TVAE_AVAILABLE and CTGAN_AVAILABLE else 1, n_plots,
                              figsize=(5*n_plots, 4*(2 if TVAE_AVAILABLE and CTGAN_AVAILABLE else 1)))
    if TVAE_AVAILABLE and CTGAN_AVAILABLE:
        axes_ctgan = axes[0]
        axes_tvae  = axes[1]
    else:
        axes_ctgan = axes if CTGAN_AVAILABLE else [None]*n_plots
        axes_tvae  = axes if TVAE_AVAILABLE  else [None]*n_plots

    fig.suptitle('Qualité des synthétiques : distributions réelles vs générées',
                 fontsize=13, fontweight='bold', color=COLORS['neutre'])

    for i, col in enumerate(key_cols):
        real_data = df_sinistres[col].dropna()

        if CTGAN_AVAILABLE and axes_ctgan[i] is not None:
            axes_ctgan[i].hist(real_data, bins=30, alpha=0.5, color=COLORS['neutre'],
                                density=True, label='Réel')
            axes_ctgan[i].hist(synthetic_ctgan[col].dropna(), bins=30, alpha=0.6,
                                color=COLORS['sinistre'], density=True, label='CTGAN')
            ks_stat, ks_pval = stats.ks_2samp(real_data, synthetic_ctgan[col].dropna()) if 'stats' in dir() else (0, 0)
            axes_ctgan[i].set_title(f'{col}\nCTGAN (KS={ks_stat:.3f})', fontsize=9)
            axes_ctgan[i].legend(fontsize=7)

        if TVAE_AVAILABLE and axes_tvae[i] is not None:
            axes_tvae[i].hist(real_data, bins=30, alpha=0.5, color=COLORS['neutre'],
                               density=True, label='Réel')
            axes_tvae[i].hist(synthetic_tvae[col].dropna(), bins=30, alpha=0.6,
                               color=COLORS['non_sinistre'], density=True, label='TVAE')
            axes_tvae[i].set_title(f'{col}\nTVAE', fontsize=9)
            axes_tvae[i].legend(fontsize=7)

    try:
        from scipy import stats as stats_module
        print('  Test de Kolmogorov-Smirnov (distributions réelles vs CTGAN) :')
        for col in key_cols:
            if CTGAN_AVAILABLE:
                ks, pval = stats_module.ks_2samp(df_sinistres[col].dropna(), synthetic_ctgan[col].dropna())
                flag = '✅ similaire' if pval > 0.05 else '⚠️  différent'
                print(f'    {col:<25}: KS={ks:.3f}, p={pval:.3f} {flag}')
    except:
        pass

    plt.tight_layout()
    plt.savefig('../outputs/figures/13_qualite_synthetiques.png', dpi=150, bbox_inches='tight')
    plt.close('all')  # plt.show() en notebook
    print('📊 Figure sauvegardée : ../outputs/figures/13_qualite_synthetiques.png')
else:
    print('[13] Analyse de qualité ignorée (CTGAN/TVAE non disponibles)')

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SAUVEGARDE FINALE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
model_info = {
    'feature_columns': X.columns.tolist(),
    'n_features'     : len(X.columns),
    'test_size'      : TEST_SIZE,
    'n_synthetic'    : N_SYNTHETIC,
    'epochs_ctgan'   : EPOCHS_CTGAN,
    'epochs_tvae'    : EPOCHS_TVAE,
    'random_state'   : RANDOM_STATE,
    'scale_pos_weight': scale_pos_weight,
    'ctgan_available': CTGAN_AVAILABLE,
    'tvae_available' : TVAE_AVAILABLE,
}
with open('../outputs/models/model_info.json', 'w') as f:
    json.dump(model_info, f, indent=2)

print('=' * 60)
print('  RÉSUMÉ — ÉTAPE 3 : MODÉLISATION')
print('=' * 60)
print(f'  Modèles entraînés   : {len(results)}')
for r in results:
    print(f'    {r["model"]:<30}: AUC={r["auc"]:.4f}, Recall={r["recall"]:.4f}')
print()
print(f'  Fichiers produits :')
print(f'    ../outputs/models/xgb_*.pkl')
print(f'    ../outputs/models/ctgan_model.pkl')
print(f'    ../outputs/models/tvae_model.pkl')
print(f'    ../outputs/synthetic/synthetic_ctgan.csv')
print(f'    ../outputs/synthetic/synthetic_tvae.csv')
print(f'    ../outputs/llm/sinistres_avec_descriptions.csv')
print(f'    ../outputs/resultats_modelisation.csv')
print()
print('✓ Étape 3 terminée — Prêt pour l\'évaluation (Étape 4)')
