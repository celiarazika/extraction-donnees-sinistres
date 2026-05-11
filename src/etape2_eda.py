"""
Étape 2 — Analyse Exploratoire (EDA)
====================================

Statistiques descriptives, distributions, corrélations,
sélection de variables. Produit 11 figures.

Auteurs : Groupe ISFA M2 2025-2026
Exécution : python etape2_eda.py

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
# # 🔍 Étape 2 — Analyse Exploratoire des Données (EDA)
#
# **Projet :** Génération de données synthétiques de sinistres pour la tarification en assurance  
# **Dataset :** Insurance Claims Data — 58 592 polices  
# **Auteurs :** Groupe ISFA 2025-2026
#
# ---
#
# ## 📋 Sommaire
# 1. Statistiques descriptives complètes
# 2. Distribution de la variable cible
# 3. Distributions des variables numériques clés
# 4. Détection des valeurs aberrantes (boxplots)
# 5. Matrice de corrélations
# 6. Corrélations avec `claim_status`
# 7. Comparaison sinistres vs non-sinistres (tests statistiques)
# 8. Taux de sinistres par variables catégorielles
# 9. Analyse des équipements de sécurité
# 10. Sélection de variables (Pearson + Spearman)
# 11. Synthèse et recommandations pour la modélisation
#
# ---
#
# ### 🎯 Objectifs
# - Comprendre la structure et les distributions des données
# - Identifier les variables les plus discriminantes pour la prédiction de sinistres
# - Valider les hypothèses actuarielles (âge, type de véhicule, densité, etc.)
# - Guider les choix de modélisation (features à conserver, transformations à appliquer)
# ───────────────────────────────────────────────────────────────────────────

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# IMPORTS & CONFIGURATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency
import os
import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11})
COLORS = {'sinistre': '#E74C3C', 'non_sinistre': '#2E86AB', 'neutre': '#1F4E79', 'accent': '#F39C12', 'light': '#ECF0F1'}

os.makedirs('../outputs/figures', exist_ok=True)

# Chargement des datasets produits en Étape 1
df         = pd.read_csv('../outputs/data_encoded.csv')
df_raw_enc = pd.read_csv('../data/Insurance claims data.csv', sep=',')

print(f'[1] Datasets chargés :')
print(f'    data_encoded.csv : {df.shape[0]:,} × {df.shape[1]}')
print(f'    Dataset brut     : {df_raw_enc.shape[0]:,} × {df_raw_enc.shape[1]}')

# Colonnes numériques originales
numeric_cols = [
    'subscription_length', 'vehicle_age', 'customer_age', 'region_density',
    'displacement', 'cylinder', 'turning_radius', 'length', 'width',
    'gross_weight', 'torque_nm', 'torque_rpm', 'power_bhp', 'power_rpm',
    'airbags', 'ncap_rating'
]
numeric_cols = [c for c in numeric_cols if c in df.columns]

df0 = df[df['claim_status'] == 0].copy()
df1 = df[df['claim_status'] == 1].copy()
print(f'    Sinistres    : {len(df1):,} ({len(df1)/len(df)*100:.1f}%)')
print(f'    Non-sinistres: {len(df0):,} ({len(df0)/len(df)*100:.1f}%)')

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. STATISTIQUES DESCRIPTIVES COMPLÈTES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print('[1] Statistiques descriptives complètes :')
desc = df[numeric_cols].describe().T.round(2)
desc['skewness'] = df[numeric_cols].skew().round(3)
desc['kurtosis'] = df[numeric_cols].kurtosis().round(3)
desc['cv_%']     = (df[numeric_cols].std() / df[numeric_cols].mean() * 100).round(1)  # Coefficient de variation

print(desc[['mean', 'std', 'cv_%', 'min', '25%', '50%', '75%', 'max', 'skewness', 'kurtosis']].to_string())
desc.to_csv('../outputs/statistiques_descriptives.csv')

print('\n  Interprétation actuarielle :')
for col in ['customer_age', 'vehicle_age', 'region_density']:
    if col in desc.index:
        sk = desc.loc[col, 'skewness']
        direction = 'droite (queue longue vers valeurs élevées)' if sk > 0.5 else 'gauche' if sk < -0.5 else 'symétrique'
        print(f'    {col}: skewness={sk} → distribution asymétrique {direction}')

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. DISTRIBUTIONS DES VARIABLES NUMÉRIQUES CLÉS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
key_vars = ['customer_age', 'vehicle_age', 'subscription_length',
            'region_density', 'power_bhp', 'torque_nm',
            'displacement', 'gross_weight']
key_vars = [c for c in key_vars if c in df.columns]

fig, axes = plt.subplots(2, 4, figsize=(18, 9))
fig.suptitle('Distributions des variables numériques clés — Sinistres vs Non-sinistres',
             fontsize=13, fontweight='bold', color=COLORS['neutre'])
axes = axes.flatten()

for i, col in enumerate(key_vars[:8]):
    axes[i].hist(df0[col].dropna(), bins=40, alpha=0.4, color=COLORS['non_sinistre'],
                 label=f'Non-sinistre (n={len(df0):,})', density=True)
    axes[i].hist(df1[col].dropna(), bins=40, alpha=0.7, color=COLORS['sinistre'],
                 label=f'Sinistre (n={len(df1):,})', density=True)

    # Lignes de moyenne
    axes[i].axvline(df0[col].mean(), color=COLORS['non_sinistre'], linewidth=2, linestyle='--', alpha=0.8)
    axes[i].axvline(df1[col].mean(), color=COLORS['sinistre'], linewidth=2, linestyle='--', alpha=0.8)

    # Différence des moyennes
    diff_pct = abs(df1[col].mean() - df0[col].mean()) / df0[col].mean() * 100 if df0[col].mean() != 0 else 0
    axes[i].set_title(f'{col}\nΔμ = {diff_pct:.1f}%', fontsize=9, fontweight='bold')
    axes[i].set_ylabel('Densité')
    axes[i].legend(fontsize=7)

plt.tight_layout()
plt.savefig('../outputs/figures/03_distributions_numeriques.png', dpi=150, bbox_inches='tight')
plt.close('all')  # plt.show() en notebook
print('📊 Figure sauvegardée : ../outputs/figures/03_distributions_numeriques.png')

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. BOXPLOTS — DÉTECTION DES VALEURS ABERRANTES PAR CLASSE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
fig.suptitle('Boxplots — Valeurs aberrantes par classe (Non-sinistre vs Sinistre)',
             fontsize=13, fontweight='bold', color=COLORS['neutre'])
axes = axes.flatten()

for i, col in enumerate(key_vars[:8]):
    data_plot = [df0[col].dropna(), df1[col].dropna()]
    bp = axes[i].boxplot(data_plot, patch_artist=True,
                          labels=['Non-sinistre', 'Sinistre'],
                          medianprops={'color': 'black', 'linewidth': 2.5},
                          flierprops={'marker': 'o', 'markersize': 2, 'alpha': 0.3})
    bp['boxes'][0].set_facecolor(COLORS['non_sinistre']); bp['boxes'][0].set_alpha(0.6)
    bp['boxes'][1].set_facecolor(COLORS['sinistre']); bp['boxes'][1].set_alpha(0.6)
    axes[i].set_title(col, fontsize=9, fontweight='bold')

    # Compter les outliers IQR pour chaque classe
    for j, data in enumerate(data_plot):
        Q1, Q3 = data.quantile(0.25), data.quantile(0.75)
        IQR = Q3 - Q1
        n_out = ((data < Q1 - 1.5*IQR) | (data > Q3 + 1.5*IQR)).sum()
        axes[i].text(j+1, data.max()*1.01, f'{n_out} out.', ha='center', fontsize=7, color='gray', style='italic')

plt.tight_layout()
plt.savefig('../outputs/figures/04_boxplots_outliers.png', dpi=150, bbox_inches='tight')
plt.close('all')  # plt.show() en notebook
print('📊 Figure sauvegardée : ../outputs/figures/04_boxplots_outliers.png')

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. MATRICE DE CORRÉLATIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
corr_data   = df[numeric_cols + ['claim_status']]
corr_matrix = corr_data.corr()

fig, ax = plt.subplots(figsize=(15, 12))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
            cmap='RdBu_r', center=0, vmin=-1, vmax=1, ax=ax,
            annot_kws={'size': 8}, linewidths=0.5, linecolor='white')
ax.set_title('Matrice de corrélations — Variables numériques + claim_status',
             fontsize=13, fontweight='bold', color=COLORS['neutre'], pad=15)

# Mettre en évidence la colonne claim_status
plt.tight_layout()
plt.savefig('../outputs/figures/05_matrice_correlations.png', dpi=150, bbox_inches='tight')
plt.close('all')  # plt.show() en notebook
print('📊 Figure sauvegardée : ../outputs/figures/05_matrice_correlations.png')

# Identifier les paires fortement corrélées (multicolinéarité potentielle)
print('\n  Paires fortement corrélées (|r| > 0.7) — risque de multicolinéarité :')
for i in range(len(corr_matrix.columns)):
    for j in range(i):
        r = corr_matrix.iloc[i, j]
        if abs(r) > 0.7 and corr_matrix.columns[i] != 'claim_status' and corr_matrix.columns[j] != 'claim_status':
            print(f'    {corr_matrix.columns[j]} ↔ {corr_matrix.columns[i]}: r={r:.3f}')

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. CORRÉLATIONS AVEC CLAIM_STATUS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
corr_target = corr_matrix['claim_status'].drop('claim_status').abs().sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(11, 7))
colors_bar = [COLORS['sinistre'] if v > 0.05 else COLORS['non_sinistre'] if v > 0.02 else '#CCCCCC'
              for v in corr_target]
bars = ax.barh(corr_target.index, corr_target.values, color=colors_bar, edgecolor='white', linewidth=1.5)
ax.axvline(x=0.05, color=COLORS['sinistre'], linestyle='--', linewidth=1.5, label='Seuil fort (0.05)', alpha=0.8)
ax.axvline(x=0.02, color=COLORS['accent'],   linestyle='--', linewidth=1.5, label='Seuil faible (0.02)', alpha=0.8)
ax.set_xlabel('Corrélation absolue avec claim_status')
ax.set_title('Corrélation des variables numériques avec claim_status',
             fontsize=13, fontweight='bold', color=COLORS['neutre'])
ax.legend()
for bar, val in zip(bars, corr_target.values):
    ax.text(val + 0.001, bar.get_y() + bar.get_height()/2, f'{val:.3f}', va='center', fontsize=8)

plt.tight_layout()
plt.savefig('../outputs/figures/06_correlations_target.png', dpi=150, bbox_inches='tight')
plt.close('all')  # plt.show() en notebook
print('📊 Figure sauvegardée : ../outputs/figures/06_correlations_target.png')
print(f'\n  Top 5 variables les plus corrélées à claim_status :')
for col, val in corr_target.sort_values(ascending=False).head(5).items():
    print(f'    {col:<25}: r={val:.4f}')

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. COMPARAISON SINISTRES VS NON-SINISTRES (TESTS STATISTIQUES)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
compare_vars = ['customer_age', 'vehicle_age', 'subscription_length', 'region_density']
compare_vars = [c for c in compare_vars if c in df.columns]

fig, axes = plt.subplots(1, len(compare_vars), figsize=(5*len(compare_vars), 6))
if len(compare_vars) == 1:
    axes = [axes]
fig.suptitle('Comparaison des distributions — Tests de Mann-Whitney',
             fontsize=13, fontweight='bold', color=COLORS['neutre'])

print('  Tests de Mann-Whitney (non-paramétrique) :')
for i, col in enumerate(compare_vars):
    stat, pval = stats.mannwhitneyu(df0[col].dropna(), df1[col].dropna(), alternative='two-sided')
    sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns'
    print(f'    {col:<25}: U={stat:.0f}, p={pval:.2e} {sig}')

    axes[i].hist(df0[col].dropna(), bins=30, alpha=0.45, color=COLORS['non_sinistre'],
                 label=f'Non-sinistre\nμ={df0[col].mean():.1f}', density=True)
    axes[i].hist(df1[col].dropna(), bins=30, alpha=0.75, color=COLORS['sinistre'],
                 label=f'Sinistre\nμ={df1[col].mean():.1f}', density=True)

    axes[i].axvline(df0[col].mean(), color=COLORS['non_sinistre'], linewidth=2, linestyle='--')
    axes[i].axvline(df1[col].mean(), color=COLORS['sinistre'],     linewidth=2, linestyle='--')

    axes[i].set_title(f'{col}\np={pval:.2e} {sig}', fontsize=9, fontweight='bold')
    axes[i].set_ylabel('Densité')
    axes[i].legend(fontsize=8)

plt.tight_layout()
plt.savefig('../outputs/figures/07_comparaison_classes.png', dpi=150, bbox_inches='tight')
plt.close('all')  # plt.show() en notebook
print('📊 Figure sauvegardée : ../outputs/figures/07_comparaison_classes.png')
print('\n  Légende : *** p<0.001 | ** p<0.01 | * p<0.05 | ns non-significatif')

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. TAUX DE SINISTRES PAR VARIABLES CATÉGORIELLES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
cat_vars = ['fuel_type', 'segment', 'transmission_type', 'rear_brakes_type', 'steering_type']
cat_vars = [c for c in cat_vars if c in df_raw_enc.columns]

fig, axes = plt.subplots(1, len(cat_vars), figsize=(5*len(cat_vars), 7))
if len(cat_vars) == 1:
    axes = [axes]
fig.suptitle('Taux de sinistres par variable catégorielle + test du χ²',
             fontsize=13, fontweight='bold', color=COLORS['neutre'])

global_rate = df_raw_enc['claim_status'].mean()
print('  Taux de sinistres par modalité :')

for i, col in enumerate(cat_vars):
    taux = df_raw_enc.groupby(col)['claim_status'].mean().sort_values(ascending=False)

    # Test du chi²
    contingency = pd.crosstab(df_raw_enc[col], df_raw_enc['claim_status'])
    chi2_stat, p_chi2, dof, _ = chi2_contingency(contingency)
    sig = '***' if p_chi2 < 0.001 else '**' if p_chi2 < 0.01 else '*' if p_chi2 < 0.05 else 'ns'

    bar_colors = [COLORS['sinistre'] if v > global_rate else COLORS['non_sinistre'] for v in taux.values]
    bars = axes[i].bar(taux.index, taux.values * 100, color=bar_colors, alpha=0.75, edgecolor='white', linewidth=1.5)
    axes[i].axhline(y=global_rate * 100, color='gray', linestyle='--', linewidth=1.5, label=f'Moyenne: {global_rate*100:.1f}%')
    axes[i].set_title(f'{col}\nχ²={chi2_stat:.1f}, p={p_chi2:.2e} {sig}', fontsize=9, fontweight='bold')
    axes[i].set_ylabel('Taux de sinistres (%)')
    axes[i].tick_params(axis='x', rotation=35)
    axes[i].legend(fontsize=7)
    for bar, val in zip(bars, taux.values):
        axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                     f'{val*100:.1f}%', ha='center', fontsize=8, fontweight='bold')

    print(f'\n    {col} (χ²={chi2_stat:.1f}, p={p_chi2:.3f} {sig}) :')
    for mod, val in taux.items():
        flag = ' ▲' if val > global_rate * 1.1 else ' ▼' if val < global_rate * 0.9 else '  '
        print(f'      {flag} {mod:<20}: {val*100:.1f}%')

plt.tight_layout()
plt.savefig('../outputs/figures/08_taux_sinistres_categorielles.png', dpi=150, bbox_inches='tight')
plt.close('all')  # plt.show() en notebook
print('\n📊 Figure sauvegardée : ../outputs/figures/08_taux_sinistres_categorielles.png')

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 8. ANALYSE DES ÉQUIPEMENTS DE SÉCURITÉ
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Amélioration : analyse de l'effet des équipements de sécurité sur la sinistralité
safety_cols = [
    'is_esc', 'is_brake_assist', 'is_tpms', 'is_parking_sensors',
    'is_parking_camera', 'is_speed_alert', 'is_ecw'
]
safety_cols = [c for c in safety_cols if c in df.columns]

if len(safety_cols) > 0:
    safety_effect = {}
    for col in safety_cols:
        with_equip    = df[df[col] == 1]['claim_status'].mean()
        without_equip = df[df[col] == 0]['claim_status'].mean()
        reduction_pct = (without_equip - with_equip) / without_equip * 100 if without_equip > 0 else 0
        safety_effect[col] = {'avec': with_equip, 'sans': without_equip, 'reduction_%': reduction_pct}

    df_safety = pd.DataFrame(safety_effect).T.sort_values('reduction_%', ascending=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(len(df_safety))
    w = 0.35
    bars1 = ax.bar([xi - w/2 for xi in x], df_safety['sans'] * 100,
                    w, label='Sans équipement', color=COLORS['sinistre'], alpha=0.7, edgecolor='white')
    bars2 = ax.bar([xi + w/2 for xi in x], df_safety['avec'] * 100,
                    w, label='Avec équipement', color=COLORS['non_sinistre'], alpha=0.7, edgecolor='white')
    ax.set_xticks(list(x))
    ax.set_xticklabels([c.replace('is_', '') for c in df_safety.index], rotation=35, ha='right')
    ax.set_ylabel('Taux de sinistres (%)')
    ax.set_title('Impact des équipements de sécurité sur le taux de sinistres',
                 fontsize=13, fontweight='bold', color=COLORS['neutre'])
    ax.legend()
    ax.axhline(global_rate * 100, color='gray', linestyle=':', linewidth=1.5, label='Moyenne globale')

    # Annotations réduction
    for i, (col, row) in enumerate(df_safety.iterrows()):
        color = COLORS['non_sinistre'] if row['reduction_%'] > 0 else COLORS['sinistre']
        ax.text(i, max(row['sans'], row['avec']) * 100 + 0.2,
                f'{row["reduction_%"]:+.1f}%', ha='center', fontsize=8,
                fontweight='bold', color=color)

    plt.tight_layout()
    plt.savefig('../outputs/figures/09_securite_sinistres.png', dpi=150, bbox_inches='tight')
    plt.close('all')  # plt.show() en notebook
    print('📊 Figure sauvegardée : ../outputs/figures/09_securite_sinistres.png')

    print('\n  Impact des équipements de sécurité :')
    print(df_safety.round(4).to_string())

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 9. SÉLECTION DE VARIABLES (Pearson + Spearman)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print('[9] Sélection de variables :')

# Pearson
corr_pearson = df[numeric_cols].corrwith(df['claim_status']).abs().sort_values(ascending=False)

# Spearman (non-linéaire)
spearman_corr = {}
for col in numeric_cols:
    rho, pval = stats.spearmanr(df[col].dropna(), df.loc[df[col].notna(), 'claim_status'])
    spearman_corr[col] = {'rho': abs(rho), 'pval': pval, 'significant': pval < 0.05}
df_spearman = pd.DataFrame(spearman_corr).T.sort_values('rho', ascending=False)

# Figure comparative
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Sélection de variables — Corrélations avec claim_status',
             fontsize=13, fontweight='bold', color=COLORS['neutre'])

corr_plot = corr_pearson.sort_values(ascending=True)
colors_p  = [COLORS['sinistre'] if v > 0.05 else COLORS['accent'] if v > 0.02 else '#DDDDDD' for v in corr_plot]
axes[0].barh(corr_plot.index, corr_plot.values, color=colors_p, edgecolor='white')
axes[0].axvline(x=0.05, color=COLORS['sinistre'], linestyle='--', linewidth=1.5, label='Seuil fort (0.05)', alpha=0.8)
axes[0].axvline(x=0.02, color=COLORS['accent'],   linestyle='--', linewidth=1.5, label='Seuil faible (0.02)', alpha=0.8)
axes[0].set_title('Corrélation de Pearson (|r|)', fontsize=11, fontweight='bold')
axes[0].set_xlabel('Corrélation absolue'); axes[0].legend()

spear_plot = df_spearman['rho'].sort_values(ascending=True)
colors_s   = [COLORS['sinistre'] if v > 0.05 else COLORS['accent'] if v > 0.02 else '#DDDDDD' for v in spear_plot]
axes[1].barh(spear_plot.index, spear_plot.values, color=colors_s, edgecolor='white')
axes[1].axvline(x=0.05, color=COLORS['sinistre'], linestyle='--', linewidth=1.5, label='Seuil fort (0.05)', alpha=0.8)
axes[1].axvline(x=0.02, color=COLORS['accent'],   linestyle='--', linewidth=1.5, label='Seuil faible (0.02)', alpha=0.8)
axes[1].set_title('Corrélation de Spearman (ρ)', fontsize=11, fontweight='bold')
axes[1].set_xlabel('Corrélation absolue'); axes[1].legend()

plt.tight_layout()
plt.savefig('../outputs/figures/10_selection_variables.png', dpi=150, bbox_inches='tight')
plt.close('all')  # plt.show() en notebook
print('📊 Figure sauvegardée : ../outputs/figures/10_selection_variables.png')

# Sauvegarde
df_selection = pd.DataFrame({'pearson_abs': corr_pearson, 'spearman_rho': df_spearman['rho'], 'spearman_pval': df_spearman['pval']}).sort_values('pearson_abs', ascending=False)
df_selection.to_csv('../outputs/selection_variables.csv')

print(f'\n  Top 8 variables discriminantes :')
for col in corr_pearson.head(8).index:
    rho = df_spearman.loc[col, 'rho'] if col in df_spearman.index else float('nan')
    print(f'    {col:<25}: Pearson={corr_pearson[col]:.4f}, Spearman={rho:.4f}')

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 10. HEATMAP TAUX DE SINISTRES — ÂGE CLIENT × ÂGE VÉHICULE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Amélioration : analyse croisée age x vehicle_age
if 'customer_age' in df_raw_enc.columns and 'vehicle_age' in df_raw_enc.columns:
    df_raw_enc['age_bin'] = pd.cut(df_raw_enc['customer_age'],
                                    bins=[17, 25, 35, 45, 55, 65, 100],
                                    labels=['18-25', '26-35', '36-45', '46-55', '56-65', '65+'])
    df_raw_enc['vehicle_age_bin'] = pd.cut(df_raw_enc['vehicle_age'],
                                            bins=[-1, 2, 5, 10, 15, 50],
                                            labels=['0-2 ans', '3-5 ans', '6-10 ans', '11-15 ans', '15+ ans'])

    pivot = df_raw_enc.pivot_table(values='claim_status', index='age_bin',
                                    columns='vehicle_age_bin', aggfunc='mean') * 100

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax,
                linewidths=0.5, linecolor='white',
                cbar_kws={'label': 'Taux de sinistres (%)'})
    ax.set_title('Taux de sinistres (%) — Âge client × Âge véhicule',
                 fontsize=13, fontweight='bold', color=COLORS['neutre'])
    ax.set_xlabel('Âge du véhicule')
    ax.set_ylabel('Âge du client')
    plt.tight_layout()
    plt.savefig('../outputs/figures/11_heatmap_age_vehicule.png', dpi=150, bbox_inches='tight')
    plt.close('all')  # plt.show() en notebook
    print('📊 Figure sauvegardée : ../outputs/figures/11_heatmap_age_vehicule.png')
    print('\n  Lecture : zones rouges = segments à forte sinistralité')

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RÉSUMÉ ÉTAPE 2
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print('=' * 65)
print('  RÉSUMÉ — ÉTAPE 2 : ANALYSE EXPLORATOIRE')
print('=' * 65)
print(f'  Dataset          : {df.shape[0]:,} polices × {df.shape[1]} colonnes')
print(f'  Taux de sinistres: {df["claim_status"].mean()*100:.1f}% (déséquilibre fort)')
print(f'  Figures générées : 11 (dans ../outputs/figures/)')
print()
print('  🔑 Insights actuariels clés :')
print(f'    → Les 5 variables les plus discriminantes :')
for col in corr_pearson.head(5).index:
    print(f'       {col} (r={corr_pearson[col]:.4f})')
print()
print('  📋 Recommandations pour la modélisation (Étape 3) :')
print('    → Conserver toutes les features (CTGAN/TVAE gère la multicolinéarité)')
print('    → Priorité métrique : Recall (coût élevé des faux négatifs en assurance)')
print('    → scale_pos_weight XGBoost = ratio imbalance ~1:14')
print('    → CTGAN/TVAE : entraîner UNIQUEMENT sur les sinistres (classe minoritaire)')
print()
print('✓ Étape 2 terminée — Prêt pour la modélisation (Étape 3)')
