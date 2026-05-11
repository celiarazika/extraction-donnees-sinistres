"""
Étape 1 — Acquisition & Prétraitement
=====================================

Charge le CSV brut, nettoie, encode, normalise.
Produit : ../outputs/data_preprocessed.csv et data_encoded.csv

Auteurs : Groupe ISFA M2 2025-2026
Exécution : python etape1_acquisition_preprocessing.py

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
# # 🏦 Étape 1 — Acquisition & Prétraitement des Données
#
# **Projet :** Génération de données synthétiques de sinistres pour la tarification en assurance  
# **Dataset :** Insurance Claims Data — 58 592 polices, 41 variables  
# **Auteurs :** Groupe ISFA 2025-2026
#
# ---
#
# ## 📋 Sommaire
# 1. Installation des dépendances
# 2. Acquisition — Chargement des données brutes
# 3. Contrôle qualité (doublons, valeurs manquantes, outliers, cohérence métier)
# 4. Suppression de l'identifiant
# 5. Parsing `max_torque` / `max_power`
# 6. Encodage binaire Yes/No
# 7. One-Hot Encoding des variables catégorielles
# 8. Analyse du déséquilibre des classes
# 9. Normalisation StandardScaler
# 10. Sauvegarde
#
# ---
#
# ### 🎯 Objectif de cette étape
# Transformer les données brutes en un dataset propre, encodé et normalisé, prêt pour l'analyse exploratoire (EDA) et la modélisation générative. La qualité du prétraitement est cruciale : **60 à 80 % du temps projet** est consacré à cette phase en pratique actuarielle.
# ───────────────────────────────────────────────────────────────────────────

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# IMPORTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Configuration graphique
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11})
COLORS = {'sinistre': '#E74C3C', 'non_sinistre': '#2E86AB', 'neutre': '#1F4E79', 'accent': '#F39C12'}

# Création des dossiers de sortie
os.makedirs('../outputs/figures', exist_ok=True)
os.makedirs('../outputs/models', exist_ok=True)
os.makedirs('../outputs/synthetic', exist_ok=True)
os.makedirs('../outputs/llm', exist_ok=True)

print('✅ Imports et configuration OK')
print(f'  Dossiers créés : ../outputs/')



DATA_PATH = '../data/Insurance claims data.csv'
print(f'✅ Dataset chargé depuis : {DATA_PATH}')

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. ACQUISITION — Chargement des données brutes
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
df_raw = pd.read_csv(DATA_PATH, sep=',')

print('=' * 60)
print('  ÉTAPE 1 — ACQUISITION & PRÉTRAITEMENT')
print('=' * 60)
print(f'\n[1] Données brutes chargées :')
print(f'    Lignes    : {df_raw.shape[0]:,}')
print(f'    Colonnes  : {df_raw.shape[1]}')
print(f'    Mémoire   : {df_raw.memory_usage(deep=True).sum() / 1024**2:.1f} MB')
print(f'\n    Types de colonnes :')
print(df_raw.dtypes.value_counts().to_string())
print(f'\n    Premières lignes :')
print(df_raw.head(3))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. CONTRÔLE QUALITÉ
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print('[2] Contrôle qualité complet :')
print('-' * 50)

# ── 2a. Doublons ────────────────────────────────────────────
n_doublons = df_raw.duplicated().sum()
print(f'\n  2a. Doublons : {n_doublons}')
if n_doublons > 0:
    df_raw = df_raw.drop_duplicates()
    print(f'      → {n_doublons} doublon(s) supprimé(s)')
else:
    print(f'      → Aucun doublon détecté ✅')

# ── 2b. Valeurs manquantes ──────────────────────────────────
n_missing = df_raw.isnull().sum()
cols_missing = n_missing[n_missing > 0]
print(f'\n  2b. Valeurs manquantes :')
if len(cols_missing) == 0:
    print(f'      → Aucune valeur manquante ✅')
else:
    print(f'      → {len(cols_missing)} colonnes avec valeurs manquantes :')
    for col, n in cols_missing.items():
        print(f'        {col:<30}: {n} ({n/len(df_raw)*100:.2f}%)')

# ── 2c. Plages de valeurs (cohérence métier) ────────────────
print(f'\n  2c. Plages de valeurs (cohérence métier) :')
checks = {
    'customer_age':        (df_raw['customer_age'].min(),        df_raw['customer_age'].max(),        'ans'),
    'vehicle_age':         (df_raw['vehicle_age'].min(),         df_raw['vehicle_age'].max(),          'ans'),
    'subscription_length': (df_raw['subscription_length'].min(), df_raw['subscription_length'].max(),  'ans'),
    'region_density':      (df_raw['region_density'].min(),      df_raw['region_density'].max(),       ''),
    'airbags':             (df_raw['airbags'].min(),             df_raw['airbags'].max(),              'unités'),
    'ncap_rating':         (df_raw['ncap_rating'].min(),         df_raw['ncap_rating'].max(),          '/5'),
}
for col, (mn, mx, unit) in checks.items():
    print(f'      {col:<25}: [{mn} – {mx}] {unit}')

# ── 2d. Assertions de cohérence métier ──────────────────────
print(f'\n  2d. Assertions métier :')
assertions = [
    ((df_raw['customer_age'] >= 18).all(), 'customer_age >= 18'),
    ((df_raw['vehicle_age'] >= 0).all(),   'vehicle_age >= 0'),
    ((df_raw['airbags'] >= 0).all(),       'airbags >= 0'),
    (df_raw['claim_status'].isin([0, 1]).all(), 'claim_status in {0, 1}'),
    ((df_raw['ncap_rating'].between(0, 5)).all(), 'ncap_rating in [0, 5]'),
]
for check, label in assertions:
    status = '✅' if check else '❌'
    print(f'      {status} {label}')

# ── 2e. Détection des outliers via IQR ──────────────────────
print(f'\n  2e. Outliers (méthode IQR — borne 1.5×IQR) :')
numeric_check = ['customer_age', 'vehicle_age', 'region_density', 'displacement', 'gross_weight', 'power_bhp' if 'power_bhp' in df_raw.columns else 'displacement']
numeric_check = [c for c in numeric_check if c in df_raw.columns]
outlier_report = {}
for col in numeric_check:
    Q1, Q3 = df_raw[col].quantile(0.25), df_raw[col].quantile(0.75)
    IQR = Q3 - Q1
    mask = (df_raw[col] < Q1 - 1.5*IQR) | (df_raw[col] > Q3 + 1.5*IQR)
    n_out = mask.sum()
    outlier_report[col] = n_out
    flag = '⚠️ ' if n_out > len(df_raw)*0.05 else '  '
    print(f'      {flag}{col:<25}: {n_out:>5} outliers ({n_out/len(df_raw)*100:.1f}%)')

print(f'\n  → Décision : outliers CONSERVÉS')
print(f'    Justification : en actuariat, les valeurs extrêmes')
print(f'    (véhicules anciens, densités élevées) sont des cas')
print(f'    réels à fort impact sur la sinistralité. Les supprimer')
print(f'    biaiserait le modèle génératif.')

# Visualisation des outliers — carte thermique des valeurs manquantes et distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Contrôle qualité — Aperçu des données brutes', fontsize=13, fontweight='bold', color=COLORS['neutre'])

# Heatmap des valeurs manquantes
missing_pct = df_raw.isnull().mean() * 100
missing_nonzero = missing_pct[missing_pct > 0]
if len(missing_nonzero) == 0:
    axes[0].text(0.5, 0.5, 'Aucune valeur\nmanquante ✅', ha='center', va='center',
                 fontsize=14, color=COLORS['neutre'], fontweight='bold',
                 transform=axes[0].transAxes)
    axes[0].set_title('Valeurs manquantes (% par colonne)')
    axes[0].axis('off')
else:
    missing_nonzero.sort_values().plot(kind='barh', ax=axes[0], color=COLORS['sinistre'])
    axes[0].set_title('Valeurs manquantes (% par colonne)')

# Distribution de la variable cible
counts = df_raw['claim_status'].value_counts()
bars = axes[1].bar(['Non-sinistre (0)', 'Sinistre (1)'], counts.values,
                    color=[COLORS['non_sinistre'], COLORS['sinistre']], edgecolor='white', linewidth=2)
axes[1].set_title('Distribution claim_status (données brutes)')
axes[1].set_ylabel('Nombre de polices')
for bar, val in zip(bars, counts.values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                 f'{val:,}\n({val/len(df_raw)*100:.1f}%)', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('../outputs/figures/00_controle_qualite.png', dpi=150, bbox_inches='tight')
plt.close('all')  # plt.show() en notebook
print('📊 Figure sauvegardée : ../outputs/figures/00_controle_qualite.png')

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. PRÉPARATION — Suppression de l'identifiant
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
df = df_raw.copy()
if 'policy_id' in df.columns:
    df = df.drop(columns=['policy_id'])
    print(f'[3] Suppression de policy_id')
    print(f'    Colonnes restantes : {df.shape[1]}')
    print(f'    Justification : identifiant unique, aucune valeur prédictive')
else:
    print(f'[3] Colonne policy_id absente — aucune suppression')
    print(f'    Colonnes : {df.shape[1]}')

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. TRANSFORMATION — Parsing max_torque et max_power
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Exemple de format brut : '113.5Nm@ 4200rpm' → torque_nm=113.5, torque_rpm=4200
if 'max_torque' in df.columns:
    print(f'[4] Parsing max_torque / max_power')
    print(f'    Exemples bruts :')
    for ex in df['max_torque'].dropna().unique()[:3]:
        print(f'      max_torque : "{ex}"')
    for ex in df['max_power'].dropna().unique()[:3]:
        print(f'      max_power  : "{ex}"')

    def parse_torque(s):
        """Extrait la valeur en Nm depuis une chaîne type '113.5Nm@ 4200rpm'."""
        m = re.search(r'([\d.]+)Nm', str(s))
        return float(m.group(1)) if m else np.nan

    def parse_torque_rpm(s):
        """Extrait le régime moteur (rpm) depuis une chaîne type '113.5Nm@ 4200rpm'."""
        m = re.search(r'@\s*([\d.]+)rpm', str(s))
        return float(m.group(1)) if m else np.nan

    def parse_power(s):
        """Extrait la puissance en BHP depuis une chaîne type '82.85bhp@ 6000rpm'."""
        m = re.search(r'([\d.]+)bhp', str(s))
        return float(m.group(1)) if m else np.nan

    def parse_power_rpm(s):
        """Extrait le régime moteur depuis max_power."""
        m = re.search(r'@\s*([\d.]+)rpm', str(s))
        return float(m.group(1)) if m else np.nan

    df['torque_nm']  = df['max_torque'].apply(parse_torque)
    df['torque_rpm'] = df['max_torque'].apply(parse_torque_rpm)
    df['power_bhp']  = df['max_power'].apply(parse_power)
    df['power_rpm']  = df['max_power'].apply(parse_power_rpm)
    df = df.drop(columns=['max_torque', 'max_power'])

    print(f'\n    4 nouvelles colonnes créées :')
    for col in ['torque_nm', 'torque_rpm', 'power_bhp', 'power_rpm']:
        nan_pct = df[col].isnull().mean() * 100
        print(f'      {col:<12}: mean={df[col].mean():.1f}, NaN={nan_pct:.1f}%')
else:
    print('[4] Colonnes max_torque/max_power absentes — parsing ignoré')

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. TRANSFORMATION — Encodage binaire Yes/No → 0/1
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
binary_cols = [
    'is_esc', 'is_adjustable_steering', 'is_tpms', 'is_parking_sensors',
    'is_parking_camera', 'is_front_fog_lights', 'is_rear_window_wiper',
    'is_rear_window_washer', 'is_rear_window_defogger', 'is_brake_assist',
    'is_power_door_locks', 'is_central_locking', 'is_power_steering',
    'is_driver_seat_height_adjustable', 'is_day_night_rear_view_mirror',
    'is_ecw', 'is_speed_alert'
]
# Ne traiter que les colonnes effectivement présentes
binary_cols = [c for c in binary_cols if c in df.columns]

print(f'[5] Encodage binaire Yes/No → 0/1')
print(f'    {len(binary_cols)} colonnes concernées')

for col in binary_cols:
    unique_vals = df[col].unique()
    df[col] = df[col].map({'Yes': 1, 'No': 0})
    if df[col].isnull().any():
        df[col] = df[col].fillna(0)
        print(f'      ⚠️  {col}: valeurs inconnues imputées à 0')

print(f'\n    Vérification post-encodage (3 colonnes) :')
for col in binary_cols[:3]:
    print(f'      {col}: {dict(df[col].value_counts().to_dict())}')

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. TRANSFORMATION — One-Hot Encoding des variables catégorielles
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
nominal_cols = ['fuel_type', 'transmission_type', 'rear_brakes_type',
                'steering_type', 'segment', 'engine_type', 'model', 'region_code']
nominal_cols = [c for c in nominal_cols if c in df.columns]

print(f'[6] One-Hot Encoding — {len(nominal_cols)} variables catégorielles')
print(f'    Modalités par variable :')
for col in nominal_cols:
    n_mod = df[col].nunique()
    print(f'      {col:<25}: {n_mod} modalités → {n_mod} colonnes dummy')

n_cols_before = df.shape[1]
df_encoded = pd.get_dummies(df, columns=nominal_cols, drop_first=False, dtype=int)
n_cols_after = df_encoded.shape[1]

print(f'\n    Dimensions avant OHE : {df.shape}')
print(f'    Dimensions après OHE : {df_encoded.shape}')
print(f'    Colonnes ajoutées    : {n_cols_after - n_cols_before}')
print(f'\n    ℹ️  drop_first=False conservé : préférable pour CTGAN/TVAE')
print(f'       qui bénéficient de la représentation complète des modalités')

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. DÉSÉQUILIBRE DES CLASSES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
n_claims    = int(df_encoded['claim_status'].sum())
n_no_claims = len(df_encoded) - n_claims
n_total     = len(df_encoded)
ratio       = n_claims / n_total * 100

print(f'[7] Analyse du déséquilibre des classes :')
print(f'    Sinistres     (1) : {n_claims:>6,} ({ratio:.1f}%)')
print(f'    Non-sinistres (0) : {n_no_claims:>6,} ({100-ratio:.1f}%)')
print(f'    Ratio imbalance   : 1 sinistre pour {n_no_claims//n_claims} non-sinistres')
print(f'')
print(f'    ⚠️  Déséquilibre FORT (ratio 1:{n_no_claims//n_claims})')
print(f'    Impact sur la modélisation :')
print(f'      → Un classificateur naïf prédisant toujours 0 aurait {100-ratio:.1f}% de précision')
print(f'      → Nécessite : SMOTE, scale_pos_weight, ou données synthétiques (CTGAN/TVAE)')
print(f'      → La métrique Recall sera prioritaire sur l\'Accuracy')

# Visualisation déséquilibre
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Analyse du déséquilibre des classes', fontsize=13, fontweight='bold', color=COLORS['neutre'])

# Camembert
wedge_props = {'edgecolor': 'white', 'linewidth': 3}
axes[0].pie([n_no_claims, n_claims],
            labels=[f'Non-sinistre (0)\n{100-ratio:.1f}%', f'Sinistre (1)\n{ratio:.1f}%'],
            colors=[COLORS['non_sinistre'], COLORS['sinistre']],
            autopct='%1.1f%%', startangle=90, wedgeprops=wedge_props,
            textprops={'fontsize': 11})
axes[0].set_title('Répartition globale')

# Barres avec annotation
bars = axes[1].bar(['Non-sinistre (0)', 'Sinistre (1)'],
                   [n_no_claims, n_claims],
                   color=[COLORS['non_sinistre'], COLORS['sinistre']],
                   edgecolor='white', linewidth=2)
axes[1].set_title('Effectifs par classe')
axes[1].set_ylabel('Nombre de polices')
for bar, val, pct in zip(bars, [n_no_claims, n_claims], [100-ratio, ratio]):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 300,
                 f'{val:,}\n({pct:.1f}%)', ha='center', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig('../outputs/figures/01_desequilibre_classes.png', dpi=150, bbox_inches='tight')
plt.close('all')  # plt.show() en notebook
print('\n📊 Figure sauvegardée : ../outputs/figures/01_desequilibre_classes.png')

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 8. NORMALISATION — StandardScaler
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
X = df_encoded.drop(columns=['claim_status'])
y = df_encoded['claim_status']

numeric_cols = [
    'subscription_length', 'vehicle_age', 'customer_age', 'region_density',
    'displacement', 'cylinder', 'turning_radius', 'length', 'width',
    'gross_weight', 'torque_nm', 'torque_rpm', 'power_bhp', 'power_rpm',
    'airbags', 'ncap_rating'
]
numeric_cols = [c for c in numeric_cols if c in X.columns]

print(f'[8] Normalisation StandardScaler')
print(f'    {len(numeric_cols)} colonnes numériques normalisées')
print(f'    Formule : z = (x - μ) / σ')
print(f'\n    Avant normalisation (3 colonnes) :')
for col in numeric_cols[:3]:
    print(f'      {col:<25}: mean={X[col].mean():.2f}, std={X[col].std():.2f}')

scaler   = StandardScaler()
X_scaled = X.copy()
X_scaled[numeric_cols] = scaler.fit_transform(X[numeric_cols])

print(f'\n    Après normalisation (3 colonnes) :')
for col in numeric_cols[:3]:
    print(f'      {col:<25}: mean={X_scaled[col].mean():.4f}, std={X_scaled[col].std():.4f}')

print(f'\n    ℹ️  StandardScaler choisi vs MinMaxScaler :')
print(f'       → Robuste aux outliers (pas de compression 0-1 forcée)')
print(f'       → Préférable pour XGBoost et réseaux de neurones (CTGAN/TVAE)')

# Visualisation avant/après normalisation
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
fig.suptitle('Normalisation StandardScaler — Avant / Après', fontsize=13, fontweight='bold', color=COLORS['neutre'])
axes = axes.flatten()

show_cols = numeric_cols[:4]
for i, col in enumerate(show_cols):
    # Avant
    axes[i].hist(X[col].dropna(), bins=40, color=COLORS['non_sinistre'], alpha=0.7, edgecolor='white')
    axes[i].set_title(f'{col}\n(brut)', fontsize=9, fontweight='bold')
    axes[i].set_ylabel('Fréquence')
    axes[i].axvline(X[col].mean(), color=COLORS['sinistre'], linewidth=2, linestyle='--', label=f'μ={X[col].mean():.1f}')
    axes[i].legend(fontsize=7)

    # Après
    axes[i+4].hist(X_scaled[col].dropna(), bins=40, color=COLORS['accent'], alpha=0.7, edgecolor='white')
    axes[i+4].set_title(f'{col}\n(normalisé)', fontsize=9, fontweight='bold')
    axes[i+4].set_ylabel('Fréquence')
    axes[i+4].axvline(0, color=COLORS['sinistre'], linewidth=2, linestyle='--', label='μ≈0')
    axes[i+4].legend(fontsize=7)

plt.tight_layout()
plt.savefig('../outputs/figures/02_normalisation.png', dpi=150, bbox_inches='tight')
plt.close('all')  # plt.show() en notebook
print('📊 Figure sauvegardée : ../outputs/figures/02_normalisation.png')

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 9. SAUVEGARDE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
import pickle

# Dataset prétraité + normalisé (pour XGBoost)
df_preprocessed = X_scaled.copy()
df_preprocessed['claim_status'] = y.values
df_preprocessed.to_csv('../outputs/data_preprocessed.csv', index=False)

# Dataset encodé sans normalisation (pour CTGAN/TVAE)
df_encoded.to_csv('../outputs/data_encoded.csv', index=False)

# Scaler sauvegardé (pour inverse_transform en étape 4)
pickle.dump(scaler, open('../outputs/models/scaler.pkl', 'wb'))
pickle.dump(numeric_cols, open('../outputs/models/numeric_cols.pkl', 'wb'))

print('[9] Fichiers sauvegardés :')
print(f'    data_preprocessed.csv : {df_preprocessed.shape[0]:,} × {df_preprocessed.shape[1]} (normalisé)')
print(f'    data_encoded.csv       : {df_encoded.shape[0]:,} × {df_encoded.shape[1]} (encodé brut)')
print(f'    scaler.pkl             : StandardScaler entraîné')
print(f'    numeric_cols.pkl       : liste des colonnes numériques')

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RÉSUMÉ ÉTAPE 1
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print('=' * 60)
print('  RÉSUMÉ — ÉTAPE 1 : ACQUISITION & PRÉTRAITEMENT')
print('=' * 60)
print(f'  Dataset brut       : {df_raw.shape[0]:,} polices × {df_raw.shape[1]} colonnes')
print(f'  Dataset encodé     : {df_encoded.shape[0]:,} polices × {df_encoded.shape[1]} colonnes')
print(f'  Dataset normalisé  : {df_preprocessed.shape[0]:,} polices × {df_preprocessed.shape[1]} colonnes')
print(f'')
print(f'  Transformations appliquées :')
print(f'    ✅ Suppression doublons')
print(f'    ✅ Contrôle cohérence métier')
print(f'    ✅ Parsing max_torque / max_power → 4 features')
print(f'    ✅ Encodage binaire {len(binary_cols)} colonnes Yes/No')
print(f'    ✅ One-Hot Encoding {len(nominal_cols)} variables catégorielles')
print(f'    ✅ StandardScaler {len(numeric_cols)} colonnes numériques')
print(f'')
print(f'  Déséquilibre des classes : 1:{n_no_claims//n_claims}')
print(f'  Stratégie de rééquilibrage : CTGAN + TVAE + SMOTE (Étape 3)')
print(f'')
print(f'✓ Étape 1 terminée — Prêt pour l\'EDA (Étape 2)')
