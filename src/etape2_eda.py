"""
Étape 2 – Analyse Exploratoire (EDA)
Projet : Génération de données synthétiques de sinistres pour la tarification
Dataset : Insurance Claims Data (58 592 polices, 41 variables)

Contenu :
  - Statistiques descriptives
  - Distribution de la variable cible
  - Distributions des variables numériques clés
  - Corrélations avec claim_status
  - Analyse par classe (sinistres vs non-sinistres)
  - Taux de sinistres par variables catégorielles
  - Détection des valeurs aberrantes (boxplots)
  - Sélection de variables (importance via corrélation et test Chi2)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
from sklearn.feature_selection import chi2, SelectKBest
import os
import warnings
warnings.filterwarnings('ignore')

# ── Configuration graphique ────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.family': 'Arial', 'font.size': 11})
COLORS = {'sinistre': '#E74C3C', 'non_sinistre': '#2E86AB', 'neutre': '#1F4E79'}

# ── Chargement ─────────────────────────────────────────────────────────────────
os.makedirs('../outputs/figures', exist_ok=True)
df = pd.read_csv('../outputs/data_encoded.csv')
print(f"[1] Dataset chargé : {df.shape[0]} lignes × {df.shape[1]} colonnes")

# Colonnes numériques originales (avant one-hot)
numeric_cols = [
    'subscription_length', 'vehicle_age', 'customer_age', 'region_density',
    'displacement', 'cylinder', 'turning_radius', 'length', 'width',
    'gross_weight', 'torque_nm', 'torque_rpm', 'power_bhp', 'power_rpm',
    'airbags', 'ncap_rating'
]

df0 = df[df['claim_status'] == 0]
df1 = df[df['claim_status'] == 1]

# ══════════════════════════════════════════════════════════════════════════════
# 1. STATISTIQUES DESCRIPTIVES
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2] Statistiques descriptives :")
desc = df[numeric_cols + ['claim_status']].describe().T
desc['skewness'] = df[numeric_cols].skew()
desc['kurtosis'] = df[numeric_cols].kurtosis()
print(desc[['mean', 'std', 'min', '25%', '50%', '75%', 'max', 'skewness', 'kurtosis']].round(2).to_string())
desc.to_csv('../outputs/statistiques_descriptives.csv')
print("    → Sauvegardé : outputs/statistiques_descriptives.csv")

# ══════════════════════════════════════════════════════════════════════════════
# 2. DISTRIBUTION DE LA VARIABLE CIBLE
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Distribution de la variable cible — claim_status", fontsize=14, fontweight='bold', color=COLORS['neutre'])

# Camembert
counts = df['claim_status'].value_counts()
axes[0].pie(
    counts, labels=['Non-sinistre (0)', 'Sinistre (1)'],
    colors=[COLORS['non_sinistre'], COLORS['sinistre']],
    autopct='%1.1f%%', startangle=90,
    wedgeprops={'edgecolor': 'white', 'linewidth': 2}
)
axes[0].set_title("Répartition globale")

# Barres
bars = axes[1].bar(['Non-sinistre (0)', 'Sinistre (1)'], counts.values,
                    color=[COLORS['non_sinistre'], COLORS['sinistre']], edgecolor='white', linewidth=1.5)
axes[1].set_title("Effectifs par classe")
axes[1].set_ylabel("Nombre de polices")
for bar, val in zip(bars, counts.values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 300,
                 f'{val:,}', ha='center', fontweight='bold')
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

plt.tight_layout()
plt.savefig('../outputs/figures/01_distribution_cible.png', dpi=150, bbox_inches='tight')
plt.close()
print("[3] Figure 1 sauvegardée : 01_distribution_cible.png")

# ══════════════════════════════════════════════════════════════════════════════
# 3. DISTRIBUTIONS DES VARIABLES NUMÉRIQUES CLÉS
# ══════════════════════════════════════════════════════════════════════════════
key_vars = ['customer_age', 'vehicle_age', 'subscription_length',
            'region_density', 'power_bhp', 'torque_nm',
            'displacement', 'gross_weight']

fig, axes = plt.subplots(2, 4, figsize=(18, 8))
fig.suptitle("Distributions des variables numériques clés", fontsize=14, fontweight='bold', color=COLORS['neutre'])
axes = axes.flatten()

for i, col in enumerate(key_vars):
    axes[i].hist(df1[col], bins=40, alpha=0.6, color=COLORS['sinistre'], label='Sinistre', density=True)
    axes[i].hist(df0[col], bins=40, alpha=0.4, color=COLORS['non_sinistre'], label='Non-sinistre', density=True)
    axes[i].set_title(col, fontsize=10, fontweight='bold')
    axes[i].set_xlabel("")
    axes[i].set_ylabel("Densité")
    axes[i].legend(fontsize=8)

plt.tight_layout()
plt.savefig('../outputs/figures/02_distributions_numeriques.png', dpi=150, bbox_inches='tight')
plt.close()
print("[4] Figure 2 sauvegardée : 02_distributions_numeriques.png")

# ══════════════════════════════════════════════════════════════════════════════
# 4. VALEURS ABERRANTES — BOXPLOTS
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
fig.suptitle("Détection des valeurs aberrantes — Boxplots par classe", fontsize=14, fontweight='bold', color=COLORS['neutre'])
axes = axes.flatten()

for i, col in enumerate(key_vars):
    data_plot = [df0[col].dropna(), df1[col].dropna()]
    bp = axes[i].boxplot(data_plot, patch_artist=True,
                          labels=['Non-sinistre', 'Sinistre'],
                          medianprops={'color': 'black', 'linewidth': 2})
    bp['boxes'][0].set_facecolor(COLORS['non_sinistre'])
    bp['boxes'][0].set_alpha(0.6)
    bp['boxes'][1].set_facecolor(COLORS['sinistre'])
    bp['boxes'][1].set_alpha(0.6)
    axes[i].set_title(col, fontsize=10, fontweight='bold')

    # Afficher le nombre d'outliers IQR
    for j, data in enumerate(data_plot):
        Q1, Q3 = data.quantile(0.25), data.quantile(0.75)
        IQR = Q3 - Q1
        n_out = ((data < Q1 - 1.5*IQR) | (data > Q3 + 1.5*IQR)).sum()
        axes[i].text(j+1, data.max()*0.98, f'{n_out} out.', ha='center', fontsize=7, color='gray')

plt.tight_layout()
plt.savefig('../outputs/figures/03_boxplots_outliers.png', dpi=150, bbox_inches='tight')
plt.close()
print("[5] Figure 3 sauvegardée : 03_boxplots_outliers.png")

# ══════════════════════════════════════════════════════════════════════════════
# 5. MATRICE DE CORRÉLATIONS
# ══════════════════════════════════════════════════════════════════════════════
corr_data = df[numeric_cols + ['claim_status']]
corr_matrix = corr_data.corr()

fig, ax = plt.subplots(figsize=(14, 11))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, ax=ax,
            annot_kws={'size': 8}, linewidths=0.5)
ax.set_title("Matrice de corrélations — variables numériques + claim_status",
             fontsize=13, fontweight='bold', color=COLORS['neutre'], pad=15)
plt.tight_layout()
plt.savefig('../outputs/figures/04_matrice_correlations.png', dpi=150, bbox_inches='tight')
plt.close()
print("[6] Figure 4 sauvegardée : 04_matrice_correlations.png")

# ══════════════════════════════════════════════════════════════════════════════
# 6. CORRÉLATIONS AVEC CLAIM_STATUS — CLASSEMENT
# ══════════════════════════════════════════════════════════════════════════════
corr_target = corr_matrix['claim_status'].drop('claim_status').abs().sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(10, 7))
colors_bar = [COLORS['sinistre'] if v > 0.05 else COLORS['non_sinistre'] for v in corr_target]
bars = ax.barh(corr_target.index, corr_target.values, color=colors_bar, edgecolor='white')
ax.axvline(x=0.05, color='gray', linestyle='--', linewidth=1, label='Seuil 0.05')
ax.set_xlabel("Corrélation absolue avec claim_status")
ax.set_title("Corrélation des variables numériques avec claim_status",
             fontsize=13, fontweight='bold', color=COLORS['neutre'])
ax.legend()
for bar, val in zip(bars, corr_target.values):
    ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=8)
plt.tight_layout()
plt.savefig('../outputs/figures/05_correlations_target.png', dpi=150, bbox_inches='tight')
plt.close()
print("[7] Figure 5 sauvegardée : 05_correlations_target.png")

# ══════════════════════════════════════════════════════════════════════════════
# 7. ANALYSE PAR CLASSE — COMPARAISON SINISTRES VS NON-SINISTRES
# ══════════════════════════════════════════════════════════════════════════════
compare_vars = ['customer_age', 'vehicle_age', 'subscription_length', 'region_density']

fig, axes = plt.subplots(1, 4, figsize=(18, 5))
fig.suptitle("Comparaison des distributions — Sinistres vs Non-sinistres", fontsize=13, fontweight='bold', color=COLORS['neutre'])

for i, col in enumerate(compare_vars):
    axes[i].hist(df0[col], bins=30, alpha=0.5, color=COLORS['non_sinistre'],
                 label=f'Non-sinistre (n={len(df0):,})', density=True)
    axes[i].hist(df1[col], bins=30, alpha=0.7, color=COLORS['sinistre'],
                 label=f'Sinistre (n={len(df1):,})', density=True)

    # Test de Mann-Whitney
    stat, pval = stats.mannwhitneyu(df0[col], df1[col], alternative='two-sided')
    sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
    axes[i].set_title(f"{col}\np-value: {pval:.2e} {sig}", fontsize=9, fontweight='bold')
    axes[i].set_ylabel("Densité")
    axes[i].legend(fontsize=7)

plt.tight_layout()
plt.savefig('../outputs/figures/06_comparaison_classes.png', dpi=150, bbox_inches='tight')
plt.close()
print("[8] Figure 6 sauvegardée : 06_comparaison_classes.png")

# ══════════════════════════════════════════════════════════════════════════════
# 8. TAUX DE SINISTRES PAR VARIABLES CATÉGORIELLES
# ══════════════════════════════════════════════════════════════════════════════
# Recharger le dataset brut encodé pour récupérer les catégorielles originales
df_raw_enc = pd.read_csv('../data/Insurance claims data.csv', sep=',')

cat_vars = ['fuel_type', 'segment', 'transmission_type', 'rear_brakes_type', 'steering_type']
fig, axes = plt.subplots(1, 5, figsize=(22, 6))
fig.suptitle("Taux de sinistres par variable catégorielle", fontsize=13, fontweight='bold', color=COLORS['neutre'])

for i, col in enumerate(cat_vars):
    taux = df_raw_enc.groupby(col)['claim_status'].mean().sort_values(ascending=False)
    bars = axes[i].bar(taux.index, taux.values * 100,
                        color=COLORS['sinistre'], alpha=0.7, edgecolor='white')
    axes[i].axhline(y=df_raw_enc['claim_status'].mean() * 100,
                     color='gray', linestyle='--', linewidth=1.5, label='Moyenne globale')
    axes[i].set_title(col, fontsize=10, fontweight='bold')
    axes[i].set_ylabel("Taux de sinistres (%)")
    axes[i].tick_params(axis='x', rotation=30)
    axes[i].legend(fontsize=7)
    for bar, val in zip(bars, taux.values):
        axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                     f'{val*100:.1f}%', ha='center', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig('../outputs/figures/07_taux_sinistres_categorielles.png', dpi=150, bbox_inches='tight')
plt.close()
print("[9] Figure 7 sauvegardée : 07_taux_sinistres_categorielles.png")

# ══════════════════════════════════════════════════════════════════════════════
# 9. SÉLECTION DE VARIABLES
# ══════════════════════════════════════════════════════════════════════════════
print("\n[10] Sélection de variables :")

# 9a. Corrélation de Pearson avec claim_status
corr_pearson = df[numeric_cols].corrwith(df['claim_status']).abs().sort_values(ascending=False)
print("\n    Top 10 variables numériques — Corrélation avec claim_status :")
print(corr_pearson.head(10).round(4).to_string())

# 9b. Test de Spearman (non-linéaire)
spearman_corr = {}
for col in numeric_cols:
    rho, pval = stats.spearmanr(df[col], df['claim_status'])
    spearman_corr[col] = {'rho': abs(rho), 'pval': pval}
df_spearman = pd.DataFrame(spearman_corr).T.sort_values('rho', ascending=False)
print("\n    Top 10 variables numériques — Corrélation de Spearman :")
print(df_spearman.head(10).round(4).to_string())

# 9c. Résumé sélection
print("\n    Synthèse sélection de variables :")
print("    Variables les plus corrélées à claim_status (Pearson > 0.02) :")
selected = corr_pearson[corr_pearson > 0.02]
for var, val in selected.items():
    print(f"      {var:<25} : {val:.4f}")

# Sauvegarde
df_selection = pd.DataFrame({
    'pearson_abs': corr_pearson,
    'spearman_rho': df_spearman['rho'],
    'spearman_pval': df_spearman['pval']
}).sort_values('pearson_abs', ascending=False)
df_selection.to_csv('../outputs/selection_variables.csv')
print("    → Sauvegardé : outputs/selection_variables.csv")

# ══════════════════════════════════════════════════════════════════════════════
# 10. FIGURE SÉLECTION DE VARIABLES
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("Sélection de variables — Corrélations avec claim_status",
             fontsize=13, fontweight='bold', color=COLORS['neutre'])

# Pearson
corr_plot = corr_pearson.sort_values(ascending=True)
colors_p = [COLORS['sinistre'] if v > 0.02 else '#CCCCCC' for v in corr_plot]
axes[0].barh(corr_plot.index, corr_plot.values, color=colors_p, edgecolor='white')
axes[0].axvline(x=0.02, color='gray', linestyle='--', linewidth=1.5, label='Seuil 0.02')
axes[0].set_title("Corrélation de Pearson (valeur absolue)", fontsize=11, fontweight='bold')
axes[0].set_xlabel("Corrélation absolue")
axes[0].legend()

# Spearman
spear_plot = df_spearman['rho'].sort_values(ascending=True)
colors_s = [COLORS['sinistre'] if v > 0.02 else '#CCCCCC' for v in spear_plot]
axes[1].barh(spear_plot.index, spear_plot.values, color=colors_s, edgecolor='white')
axes[1].axvline(x=0.02, color='gray', linestyle='--', linewidth=1.5, label='Seuil 0.02')
axes[1].set_title("Corrélation de Spearman (valeur absolue)", fontsize=11, fontweight='bold')
axes[1].set_xlabel("Corrélation absolue")
axes[1].legend()

plt.tight_layout()
plt.savefig('../outputs/figures/08_selection_variables.png', dpi=150, bbox_inches='tight')
plt.close()
print("[11] Figure 8 sauvegardée : 08_selection_variables.png")

# ══════════════════════════════════════════════════════════════════════════════
# RÉSUMÉ FINAL
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  RÉSUMÉ ÉTAPE 2 — ANALYSE EXPLORATOIRE")
print("="*60)
print(f"  Dataset          : {df.shape[0]:,} polices × {df.shape[1]} colonnes")
print(f"  Taux de sinistres: {df['claim_status'].mean()*100:.1f}%")
print(f"  Figures générées : 8 (dans outputs/figures/)")
print(f"  Fichiers CSV     : statistiques_descriptives.csv, selection_variables.csv")
print(f"\n  Variables les plus discriminantes :")
for var in corr_pearson.head(5).index:
    print(f"    → {var} (r={corr_pearson[var]:.4f})")
print("\n✓ Étape 2 terminée.")