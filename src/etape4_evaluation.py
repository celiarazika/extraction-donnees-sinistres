"""
Étape 4 – Évaluation
Projet : Génération de données synthétiques de sinistres pour la tarification

Contenu :
  A. Évaluation supervisée     — F1, AUC, RMSE, Precision, Recall
  B. Évaluation non supervisée — Silhouette, Perplexité, KS test, TSTR, PCA
  C. Analyse de sensibilité    — Seuil, Volume, Hyperparamètres, Variables
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, f1_score, recall_score,
                             precision_score, mean_squared_error,
                             confusion_matrix, roc_curve,
                             precision_recall_curve, silhouette_score)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from scipy import stats
import xgboost as xgb

os.makedirs('../outputs/figures', exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.family': 'Arial', 'font.size': 11})
COLORS = {'sinistre': '#E74C3C', 'non_sinistre': '#2E86AB',
          'neutre': '#1F4E79', 'ctgan': '#2ECC71', 'tvae': '#F39C12'}

RANDOM_STATE = 42
TEST_SIZE    = 0.2

print("=" * 65)
print("  ÉTAPE 4 — ÉVALUATION COMPLÈTE")
print("=" * 65)

# ── Chargement ─────────────────────────────────────────────────────────────────
df_encoded = pd.read_csv('../outputs/data_encoded.csv')
df_pre     = pd.read_csv('../outputs/data_preprocessed.csv')
results_df = pd.read_csv('../outputs/resultats_modelisation.csv')

X = df_pre.drop(columns=['claim_status'])
y = df_pre['claim_status']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

xgb_baseline    = pickle.load(open('../outputs/models/xgb_baseline.pkl', 'rb'))
synthetic_ctgan = pd.read_csv('../outputs/synthetic/synthetic_ctgan.csv')
synthetic_tvae  = pd.read_csv('../outputs/synthetic/synthetic_tvae.csv')
df_sinistres    = df_encoded[df_encoded['claim_status'] == 1].copy()

numeric_cols = ['subscription_length','vehicle_age','customer_age','region_density',
                'displacement','cylinder','gross_weight','torque_nm','power_bhp','airbags']

print(f"[✓] Données et modèles chargés")

# ══════════════════════════════════════════════════════════════════════════════
# A. ÉVALUATION SUPERVISÉE
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("  A. ÉVALUATION SUPERVISÉE")
print(f"{'='*65}")

model_files = {
    'XGBoost Baseline': '../outputs/models/xgb_baseline.pkl',
    'XGBoost + CTGAN' : '../outputs/models/xgb_ctgan.pkl',
    'XGBoost + TVAE'  : '../outputs/models/xgb_tvae.pkl',
    'XGBoost + SMOTE' : '../outputs/models/xgb_smote.pkl',
}
model_colors = ['#1F4E79','#2ECC71','#F39C12','#E74C3C']

# A1. Métriques complètes incluant RMSE
print("\n[A1] Métriques supervisées complètes :")
print(f"\n  {'Modèle':<28} {'AUC':>7} {'F1':>7} {'Recall':>7} {'Precision':>10} {'RMSE':>7}")
print("  " + "-"*65)

sup_results = []
for name, path in model_files.items():
    try:
        model    = pickle.load(open(path, 'rb'))
        y_pred   = model.predict(X_test)
        y_prob   = model.predict_proba(X_test)[:, 1]
        auc      = roc_auc_score(y_test, y_prob)
        f1       = f1_score(y_test, y_pred)
        rec      = recall_score(y_test, y_pred)
        prec     = precision_score(y_test, y_pred, zero_division=0)
        rmse     = np.sqrt(mean_squared_error(y_test, y_prob))
        print(f"  {name:<28} {auc:>7.4f} {f1:>7.4f} {rec:>7.4f} {prec:>10.4f} {rmse:>7.4f}")
        sup_results.append({'model': name, 'auc': auc, 'f1': f1,
                            'recall': rec, 'precision': prec, 'rmse': rmse})
    except Exception as e:
        print(f"  {name:<28} Erreur : {e}")

df_sup = pd.DataFrame(sup_results)
df_sup.to_csv('../outputs/evaluation_supervisee.csv', index=False)

# A2. Courbes ROC et PR
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Évaluation supervisée — Courbes ROC et Précision-Rappel",
             fontsize=13, fontweight='bold', color=COLORS['neutre'])

for (name, path), color in zip(model_files.items(), model_colors):
    try:
        model  = pickle.load(open(path, 'rb'))
        y_prob = model.predict_proba(X_test)[:, 1]
        auc    = roc_auc_score(y_test, y_prob)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        prec_c, rec_c, _ = precision_recall_curve(y_test, y_prob)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})", color=color, linewidth=2)
        axes[1].plot(rec_c, prec_c, label=name, color=color, linewidth=2)
    except:
        pass

axes[0].plot([0,1],[0,1],'k--',linewidth=1,label='Hasard (0.5)')
axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
axes[0].set_title("Courbe ROC", fontweight='bold'); axes[0].legend(fontsize=8)

axes[1].axhline(y=y_test.mean(), color='gray', linestyle='--', linewidth=1)
axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
axes[1].set_title("Courbe Précision-Rappel", fontweight='bold'); axes[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig('../outputs/figures/10_courbes_roc_pr.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n    → Sauvegardé : 10_courbes_roc_pr.png")
print(f"    → Sauvegardé : outputs/evaluation_supervisee.csv")

# ══════════════════════════════════════════════════════════════════════════════
# B. ÉVALUATION NON SUPERVISÉE
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("  B. ÉVALUATION NON SUPERVISÉE")
print(f"{'='*65}")

available_cols = [c for c in numeric_cols if c in synthetic_ctgan.columns and c in df_sinistres.columns]

# ── Préparation des échantillons ───────────────────────────────────────────────
n_sample     = min(500, len(df_sinistres), len(synthetic_ctgan), len(synthetic_tvae))
real_sample  = df_sinistres[available_cols].dropna().sample(n_sample, random_state=RANDOM_STATE)
ctgan_sample = synthetic_ctgan[available_cols].dropna().sample(n_sample, random_state=RANDOM_STATE)
tvae_sample  = synthetic_tvae[available_cols].dropna().sample(n_sample, random_state=RANDOM_STATE)

scaler = StandardScaler()
real_scaled  = scaler.fit_transform(real_sample)
ctgan_scaled = scaler.transform(ctgan_sample)
tvae_scaled  = scaler.transform(tvae_sample)

# B1. SCORE DE SILHOUETTE
print("\n[B1] Score de Silhouette — séparabilité réel vs synthétique")
print("    Principe : un score proche de 0 indique que les synthétiques")
print("    sont bien mélangés aux réels (non distinguables) → bonne qualité.")
print("    Un score proche de 1 indique qu'ils sont séparables → mauvaise qualité.")

for name, scaled in [('CTGAN', ctgan_scaled), ('TVAE', tvae_scaled)]:
    X_sil    = np.vstack([real_scaled, scaled])
    y_sil    = np.array([0]*n_sample + [1]*n_sample)  # 0=réel, 1=synthétique
    sil_score = silhouette_score(X_sil, y_sil, sample_size=min(1000, len(X_sil)),
                                  random_state=RANDOM_STATE)
    print(f"\n    Silhouette {name} : {sil_score:.4f}")
    if abs(sil_score) < 0.1:
        print(f"    → Excellent : réels et synthétiques {name} quasi-indistinguables")
    elif abs(sil_score) < 0.3:
        print(f"    → Bon : légère séparabilité résiduelle entre réels et {name}")
    else:
        print(f"    → Insuffisant : {name} trop séparable des données réelles")

# B2. PERPLEXITÉ — via modèle discriminateur
print("\n[B2] Perplexité — via modèle discriminateur (Détection synthétique)")
print("    Principe : entraîner un classifieur à distinguer réel vs synthétique.")
print("    Si l'accuracy est proche de 50% → les synthétiques sont indistinguables.")
print("    Si l'accuracy est proche de 100% → les synthétiques sont trop différents.")

perp_results = {}
for name, scaled in [('CTGAN', ctgan_scaled), ('TVAE', tvae_scaled)]:
    X_disc = np.vstack([real_scaled, scaled])
    y_disc = np.array([0]*n_sample + [1]*n_sample)

    # Split du jeu discriminateur
    X_d_tr, X_d_te, y_d_tr, y_d_te = train_test_split(
        X_disc, y_disc, test_size=0.3, random_state=RANDOM_STATE, stratify=y_disc)

    disc = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    disc.fit(X_d_tr, y_d_tr)
    acc_disc = disc.score(X_d_te, y_d_te)

    # Perplexité approchée = entropie des probabilités
    probs    = disc.predict_proba(X_d_te)
    probs    = np.clip(probs, 1e-10, 1)
    entropy  = -np.mean(np.sum(probs * np.log2(probs), axis=1))
    perp     = 2 ** entropy

    perp_results[name] = {'accuracy': acc_disc, 'perplexite': perp}
    print(f"\n    {name} :")
    print(f"      Accuracy discriminateur : {acc_disc:.4f}")
    print(f"      Perplexité              : {perp:.4f}")
    if acc_disc < 0.6:
        print(f"      → Excellent : le discriminateur ne distingue pas les {name}")
    elif acc_disc < 0.75:
        print(f"      → Acceptable : légère différence détectable")
    else:
        print(f"      → Insuffisant : {name} trop différents des réels")

# B3. TEST KS — fidélité des distributions
print("\n[B3] Test de Kolmogorov-Smirnov (réel vs synthétique)")
print(f"\n  {'Variable':<25} {'KS CTGAN':>10} {'p CTGAN':>10} {'KS TVAE':>10} {'p TVAE':>10}")
print("  " + "-"*68)

ks_results = []
for col in available_cols:
    real_v  = df_sinistres[col].dropna().values
    ctgan_v = synthetic_ctgan[col].dropna().values
    tvae_v  = synthetic_tvae[col].dropna().values
    ks_c, p_c = stats.ks_2samp(real_v, ctgan_v)
    ks_t, p_t = stats.ks_2samp(real_v, tvae_v)
    print(f"  {col:<25} {ks_c:>10.4f} {p_c:>10.4f} {ks_t:>10.4f} {p_t:>10.4f}")
    ks_results.append({'variable': col, 'ks_ctgan': ks_c, 'p_ctgan': p_c,
                        'ks_tvae': ks_t, 'p_tvae': p_t})

df_ks = pd.DataFrame(ks_results)
df_ks.to_csv('../outputs/ks_test_results.csv', index=False)
print("\n    Interprétation : KS proche de 0 et p > 0.05 → distributions similaires")
print("    → Sauvegardé : outputs/ks_test_results.csv")

# B4. TSTR
print("\n[B4] TSTR — Train on Synthetic, Test on Real")
tstr_results = {}
for name, synthetic in [('CTGAN', synthetic_ctgan), ('TVAE', synthetic_tvae)]:
    try:
        syn_X = synthetic.drop(columns=['claim_status'], errors='ignore')
        syn_X = syn_X.reindex(columns=X_train.columns, fill_value=0)
        n_neg = len(syn_X)
        X_neg = X_train[y_train==0].sample(n=n_neg, random_state=RANDOM_STATE)
        X_tstr = pd.concat([syn_X, X_neg], ignore_index=True)
        y_tstr = pd.concat([pd.Series([1]*n_neg), pd.Series([0]*n_neg)], ignore_index=True)
        m = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
                               scale_pos_weight=1, random_state=RANDOM_STATE,
                               eval_metric='auc', verbosity=0)
        m.fit(X_tstr, y_tstr, verbose=False)
        auc_t = roc_auc_score(y_test, m.predict_proba(X_test)[:,1])
        f1_t  = f1_score(y_test, m.predict(X_test))
        tstr_results[name] = {'auc': auc_t, 'f1': f1_t}
        print(f"    TSTR {name} : AUC={auc_t:.4f}, F1={f1_t:.4f}")
    except Exception as e:
        print(f"    ⚠️  TSTR {name} : {e}")

auc_base = results_df[results_df['model']=='XGBoost Baseline']['auc'].values[0]
print(f"    TRTR Baseline  : AUC={auc_base:.4f}")

# B5. Figure non supervisée
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Évaluation non supervisée — Qualité des données synthétiques",
             fontsize=13, fontweight='bold', color=COLORS['neutre'])

# PCA
all_data   = np.vstack([real_scaled, ctgan_scaled, tvae_scaled])
pca        = PCA(n_components=2, random_state=RANDOM_STATE)
all_pca    = pca.fit_transform(all_data)
axes[0,0].scatter(all_pca[:n_sample,0], all_pca[:n_sample,1],
                   c=COLORS['sinistre'], alpha=0.4, s=15, label='Réels')
axes[0,0].scatter(all_pca[n_sample:2*n_sample,0], all_pca[n_sample:2*n_sample,1],
                   c=COLORS['ctgan'], alpha=0.4, s=15, label='CTGAN')
axes[0,0].scatter(all_pca[2*n_sample:,0], all_pca[2*n_sample:,1],
                   c=COLORS['tvae'], alpha=0.4, s=15, label='TVAE')
axes[0,0].set_title(f"PCA — Réel vs Synthétiques\n(PC1={pca.explained_variance_ratio_[0]*100:.1f}%, PC2={pca.explained_variance_ratio_[1]*100:.1f}%)", fontweight='bold')
axes[0,0].legend()

# KS scores
df_ks_p = df_ks.dropna()
x = np.arange(len(df_ks_p)); w = 0.35
axes[0,1].bar(x-w/2, df_ks_p['ks_ctgan'], w, label='CTGAN', color=COLORS['ctgan'], alpha=0.8)
axes[0,1].bar(x+w/2, df_ks_p['ks_tvae'],  w, label='TVAE',  color=COLORS['tvae'],  alpha=0.8)
axes[0,1].axhline(y=0.1, color='gray', linestyle='--', linewidth=1.5, label='Seuil 0.1')
axes[0,1].set_xticks(x)
axes[0,1].set_xticklabels(df_ks_p['variable'], rotation=30, ha='right', fontsize=8)
axes[0,1].set_ylabel("Statistique KS (↓ mieux)")
axes[0,1].set_title("Test KS — Fidélité des distributions", fontweight='bold')
axes[0,1].legend()

# Silhouette et Perplexité
names_p  = ['CTGAN', 'TVAE']
sil_vals = []
for name, scaled in [('CTGAN', ctgan_scaled), ('TVAE', tvae_scaled)]:
    X_s = np.vstack([real_scaled, scaled])
    y_s = np.array([0]*n_sample + [1]*n_sample)
    sil_vals.append(abs(silhouette_score(X_s, y_s, sample_size=min(1000, len(X_s)),
                                          random_state=RANDOM_STATE)))

bars = axes[1,0].bar(names_p, sil_vals, color=[COLORS['ctgan'], COLORS['tvae']], alpha=0.8, edgecolor='white')
axes[1,0].axhline(y=0.1, color='gray', linestyle='--', linewidth=1.5, label='Seuil 0.1 (bon)')
axes[1,0].set_ylabel("Score de Silhouette |s| (↓ mieux)")
axes[1,0].set_title("Score de Silhouette\n(0 = indistinguable des réels)", fontweight='bold')
axes[1,0].legend()
for bar, val in zip(bars, sil_vals):
    axes[1,0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
                   f'{val:.4f}', ha='center', fontweight='bold')

# Accuracy discriminateur
disc_acc = [perp_results.get('CTGAN',{}).get('accuracy', 0),
            perp_results.get('TVAE',{}).get('accuracy', 0)]
bars2 = axes[1,1].bar(names_p, disc_acc, color=[COLORS['ctgan'], COLORS['tvae']], alpha=0.8, edgecolor='white')
axes[1,1].axhline(y=0.5, color='gray', linestyle='--', linewidth=1.5, label='Hasard (50%)')
axes[1,1].axhline(y=0.6, color='orange', linestyle=':', linewidth=1.5, label='Seuil acceptable (60%)')
axes[1,1].set_ylim(0, 1)
axes[1,1].set_ylabel("Accuracy du discriminateur (↓ mieux)")
axes[1,1].set_title("Perplexité — Détection synthétique\n(50% = indistinguable)", fontweight='bold')
axes[1,1].legend(fontsize=8)
for bar, val in zip(bars2, disc_acc):
    axes[1,1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                   f'{val:.4f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('../outputs/figures/11_evaluation_non_supervisee.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n    → Sauvegardé : 11_evaluation_non_supervisee.png")

# ══════════════════════════════════════════════════════════════════════════════
# C. ANALYSE DE SENSIBILITÉ
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("  C. ANALYSE DE SENSIBILITÉ")
print(f"{'='*65}")

# C1. Seuil de décision
print("\n[C1] Sensibilité au seuil de décision")
y_prob_base  = xgb_baseline.predict_proba(X_test)[:,1]
thresholds   = np.arange(0.05, 0.95, 0.05)
thresh_res   = []
for t in thresholds:
    y_t = (y_prob_base >= t).astype(int)
    thresh_res.append({
        'threshold': t,
        'f1'       : f1_score(y_test, y_t, zero_division=0),
        'recall'   : recall_score(y_test, y_t, zero_division=0),
        'precision': precision_score(y_test, y_t, zero_division=0)
    })
df_thresh     = pd.DataFrame(thresh_res)
best_thresh   = df_thresh.loc[df_thresh['f1'].idxmax(), 'threshold']
print(f"    Seuil optimal (F1 max) : {best_thresh:.2f}")
print(f"    F1 au seuil optimal    : {df_thresh['f1'].max():.4f}")
print(f"    Recall au seuil optimal: {df_thresh.loc[df_thresh['f1'].idxmax(),'recall']:.4f}")

fig, ax = plt.subplots(figsize=(10,5))
ax.plot(df_thresh['threshold'], df_thresh['f1'],       label='F1-score',  color=COLORS['neutre'],       linewidth=2.5)
ax.plot(df_thresh['threshold'], df_thresh['recall'],   label='Recall',    color=COLORS['sinistre'],      linewidth=2.5)
ax.plot(df_thresh['threshold'], df_thresh['precision'],label='Precision', color=COLORS['non_sinistre'], linewidth=2.5)
ax.axvline(x=best_thresh, color='gray', linestyle='--', linewidth=1.5, label=f'Seuil optimal ({best_thresh:.2f})')
ax.set_xlabel("Seuil de décision"); ax.set_ylabel("Score")
ax.set_title("Sensibilité au seuil de décision — XGBoost Baseline",
             fontweight='bold', color=COLORS['neutre'])
ax.legend()
plt.tight_layout()
plt.savefig('../outputs/figures/12_sensibilite_seuil.png', dpi=150, bbox_inches='tight')
plt.close()
print("    → Sauvegardé : 12_sensibilite_seuil.png")

# C2. Volume de données synthétiques
print("\n[C2] Sensibilité au volume synthétique (CTGAN)")
volumes    = [500, 1000, 2000, 3000, 5000]
vol_res    = []
for n in volumes:
    try:
        syn_s = synthetic_ctgan.sample(min(n, len(synthetic_ctgan)), random_state=RANDOM_STATE)
        syn_X = syn_s.drop(columns=['claim_status'], errors='ignore').reindex(columns=X_train.columns, fill_value=0)
        X_aug = pd.concat([X_train, syn_X], ignore_index=True)
        y_aug = pd.concat([y_train, pd.Series([1]*len(syn_X))], ignore_index=True)
        spw   = max(1, int((y_aug==0).sum()/(y_aug==1).sum()))
        m     = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
                                   scale_pos_weight=spw, random_state=RANDOM_STATE,
                                   eval_metric='auc', verbosity=0)
        m.fit(X_aug, y_aug, verbose=False)
        auc_v = roc_auc_score(y_test, m.predict_proba(X_test)[:,1])
        f1_v  = f1_score(y_test, m.predict(X_test))
        vol_res.append({'n_synthetic': n, 'auc': auc_v, 'f1': f1_v})
        print(f"    N={n:>5} : AUC={auc_v:.4f}, F1={f1_v:.4f}")
    except Exception as e:
        print(f"    N={n:>5} : Erreur — {e}")

df_vol = pd.DataFrame(vol_res)
df_vol.to_csv('../outputs/sensibilite_volume.csv', index=False)

fig, axes = plt.subplots(1,2, figsize=(12,5))
fig.suptitle("Sensibilité au volume de données synthétiques CTGAN",
             fontsize=13, fontweight='bold', color=COLORS['neutre'])
if len(df_vol) > 0:
    axes[0].plot(df_vol['n_synthetic'], df_vol['auc'], 'o-', color=COLORS['neutre'], linewidth=2.5, markersize=8)
    axes[0].axhline(y=auc_base, color='gray', linestyle='--', linewidth=1.5, label=f'Baseline ({auc_base:.4f})')
    axes[0].set_xlabel("N synthétiques"); axes[0].set_ylabel("AUC-ROC")
    axes[0].set_title("AUC-ROC vs volume", fontweight='bold'); axes[0].legend()
    f1_base = results_df[results_df['model']=='XGBoost Baseline']['f1'].values[0]
    axes[1].plot(df_vol['n_synthetic'], df_vol['f1'], 'o-', color=COLORS['sinistre'], linewidth=2.5, markersize=8)
    axes[1].axhline(y=f1_base, color='gray', linestyle='--', linewidth=1.5, label=f'Baseline ({f1_base:.4f})')
    axes[1].set_xlabel("N synthétiques"); axes[1].set_ylabel("F1-score")
    axes[1].set_title("F1-score vs volume", fontweight='bold'); axes[1].legend()
plt.tight_layout()
plt.savefig('../outputs/figures/13_sensibilite_volume.png', dpi=150, bbox_inches='tight')
plt.close()
print("    → Sauvegardé : 13_sensibilite_volume.png")

# C3. Hyperparamètres
print("\n[C3] Grille max_depth × learning_rate")
depths  = [3,4,6,8]
lr_vals = [0.01,0.05,0.1,0.2]
hp_auc  = np.zeros((len(depths), len(lr_vals)))
for i, depth in enumerate(depths):
    for j, lr in enumerate(lr_vals):
        m = xgb.XGBClassifier(n_estimators=200, max_depth=depth, learning_rate=lr,
                               subsample=0.8, colsample_bytree=0.8,
                               scale_pos_weight=14, random_state=RANDOM_STATE,
                               eval_metric='auc', verbosity=0)
        m.fit(X_train, y_train, verbose=False)
        hp_auc[i,j] = roc_auc_score(y_test, m.predict_proba(X_test)[:,1])
        print(f"    depth={depth}, lr={lr} → AUC={hp_auc[i,j]:.4f}")

fig, ax = plt.subplots(figsize=(9,6))
df_hp = pd.DataFrame(hp_auc, index=[f"depth={d}" for d in depths],
                      columns=[f"lr={lr}" for lr in lr_vals])
sns.heatmap(df_hp, annot=True, fmt='.4f', cmap='YlOrRd', ax=ax,
            annot_kws={'size':10}, linewidths=0.5)
ax.set_title("AUC-ROC — Sensibilité max_depth × learning_rate",
             fontweight='bold', color=COLORS['neutre'])
plt.tight_layout()
plt.savefig('../outputs/figures/14_sensibilite_hyperparametres.png', dpi=150, bbox_inches='tight')
plt.close()
print("    → Sauvegardé : 14_sensibilite_hyperparametres.png")

# C4. Modèle réduit vs complet
print("\n[C4] Modèle réduit (5 variables) vs modèle complet")
selected_vars = ['subscription_length','vehicle_age','customer_age','region_density','cylinder']
m_red = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8,
                            scale_pos_weight=14, random_state=RANDOM_STATE,
                            eval_metric='auc', verbosity=0)
m_red.fit(X_train[selected_vars], y_train, verbose=False)
auc_red = roc_auc_score(y_test, m_red.predict_proba(X_test[selected_vars])[:,1])
f1_red  = f1_score(y_test, m_red.predict(X_test[selected_vars]))
rec_red = recall_score(y_test, m_red.predict(X_test[selected_vars]))
print(f"    Réduit (5 vars)  : AUC={auc_red:.4f}, F1={f1_red:.4f}, Recall={rec_red:.4f}")
print(f"    Complet (93 vars): AUC={auc_base:.4f}, F1={results_df[results_df['model']=='XGBoost Baseline']['f1'].values[0]:.4f}")
print(f"    Perte AUC        : {auc_base-auc_red:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# RÉSUMÉ FINAL
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("  RÉSUMÉ ÉTAPE 4")
print(f"{'='*65}")
print(f"\n  A. Supervisée")
print(f"     Meilleur AUC    : {df_sup['auc'].max():.4f} ({df_sup.loc[df_sup['auc'].idxmax(),'model']})")
print(f"     Meilleur F1     : {df_sup['f1'].max():.4f} ({df_sup.loc[df_sup['f1'].idxmax(),'model']})")
print(f"     Meilleur Recall : {df_sup['recall'].max():.4f} ({df_sup.loc[df_sup['recall'].idxmax(),'model']})")
print(f"\n  B. Non supervisée")
for name in ['CTGAN','TVAE']:
    if name in perp_results:
        print(f"     Discriminateur {name} : accuracy={perp_results[name]['accuracy']:.4f}, perplexité={perp_results[name]['perplexite']:.4f}")
for name, res in tstr_results.items():
    print(f"     TSTR {name}         : AUC={res['auc']:.4f}")
print(f"\n  C. Sensibilité")
print(f"     Seuil optimal (F1) : {best_thresh:.2f} → F1={df_thresh['f1'].max():.4f}")
print(f"     Modèle réduit      : AUC={auc_red:.4f} (perte={auc_base-auc_red:.4f})")
print(f"\n  Figures : 10 à 14 dans outputs/figures/")
print(f"\n✓ Étape 4 terminée.")
