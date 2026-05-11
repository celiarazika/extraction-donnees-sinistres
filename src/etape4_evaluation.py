"""
Étape 4 — Évaluation Complète (VERSION AMÉLIORÉE)
=================================================

Projet : Génération de données synthétiques de sinistres pour la tarification

Contenu :
  A. Évaluation supervisée     — F1, AUC, RMSE, Precision, Recall, ROC, PR
  B. Évaluation non supervisée — Silhouette, Perplexité, KS, TSTR, PCA,
                                  K-means clustering, t-SNE (AJOUTÉS)
  C. Analyse de sensibilité    — Seuil, Volume, Hyperparamètres, Variables
  D. LLM-as-a-Judge            — Évaluation qualitative des descriptions LLM (AJOUTÉ)
  E. Calibration probabiliste  — Brier score + reliability diagram (AJOUTÉ)

Auteurs : Groupe ISFA M2 2025-2026
Exécution : python etape4_evaluation.py
"""

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (roc_auc_score, f1_score, recall_score,
                             precision_score, mean_squared_error,
                             confusion_matrix, roc_curve,
                             precision_recall_curve, silhouette_score,
                             davies_bouldin_score, calinski_harabasz_score,
                             brier_score_loss)
from sklearn.calibration import calibration_curve
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from scipy import stats
import xgboost as xgb

os.makedirs('../outputs/figures', exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11})
COLORS = {'sinistre': '#E74C3C', 'non_sinistre': '#2E86AB',
          'neutre': '#1F4E79', 'ctgan': '#2ECC71', 'tvae': '#F39C12',
          'vert': '#27AE60', 'gris': '#95A5A6'}

RANDOM_STATE = 42
TEST_SIZE    = 0.2

print("=" * 70)
print("  ÉTAPE 4 — ÉVALUATION COMPLÈTE (VERSION AMÉLIORÉE)")
print("=" * 70)

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
print(f"\n{'='*70}")
print("  A. ÉVALUATION SUPERVISÉE")
print(f"{'='*70}")

model_files = {
    'XGBoost Baseline': '../outputs/models/xgb_baseline.pkl',
    'XGBoost + CTGAN' : '../outputs/models/xgb_ctgan.pkl',
    'XGBoost + TVAE'  : '../outputs/models/xgb_tvae.pkl',
    'XGBoost + SMOTE' : '../outputs/models/xgb_smote.pkl',
}
model_colors = ['#1F4E79','#2ECC71','#F39C12','#E74C3C']

# A1. Métriques complètes incluant RMSE
print("\n[A1] Métriques supervisées détaillées")
sup_metrics = []
for name, path in model_files.items():
    if not os.path.exists(path):
        continue
    m = pickle.load(open(path, 'rb'))
    y_pred = m.predict(X_test)
    y_prob = m.predict_proba(X_test)[:, 1]
    sup_metrics.append({
        'model'    : name,
        'auc'      : roc_auc_score(y_test, y_prob),
        'f1'       : f1_score(y_test, y_pred, zero_division=0),
        'recall'   : recall_score(y_test, y_pred, zero_division=0),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'rmse'     : np.sqrt(mean_squared_error(y_test, y_prob)),
        'brier'    : brier_score_loss(y_test, y_prob),
    })
df_sup = pd.DataFrame(sup_metrics)
print(df_sup.to_string(index=False))
df_sup.to_csv('../outputs/metriques_supervisees.csv', index=False)

# A2. Courbes ROC et PR
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Comparaison des modèles — Courbes ROC et Précision-Rappel",
             fontsize=13, fontweight='bold', color=COLORS['neutre'])
for (name, path), color in zip(model_files.items(), model_colors):
    if not os.path.exists(path):
        continue
    m = pickle.load(open(path, 'rb'))
    y_prob = m.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    axes[0].plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})", linewidth=2.5, color=color)
    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    axes[1].plot(rec, prec, label=name, linewidth=2.5, color=color)

axes[0].plot([0,1],[0,1],'--',color='gray', linewidth=1, alpha=0.5)
axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("Courbe ROC", fontweight='bold'); axes[0].legend(fontsize=9)
axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
axes[1].set_title("Courbe Précision-Rappel", fontweight='bold'); axes[1].legend(fontsize=9)
plt.tight_layout()
plt.savefig('../outputs/figures/10_roc_pr_supervise.png', dpi=150, bbox_inches='tight')
plt.close()
print("    → Sauvegardé : 10_roc_pr_supervise.png")

# ══════════════════════════════════════════════════════════════════════════════
# B. ÉVALUATION NON SUPERVISÉE
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("  B. ÉVALUATION NON SUPERVISÉE")
print(f"{'='*70}")

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

# ── B1. SCORE DE SILHOUETTE ────────────────────────────────────────────────────
print("\n[B1] Score de Silhouette — séparabilité réel vs synthétique")
print("    Principe : un score proche de 0 indique que les synthétiques")
print("    sont bien mélangés aux réels (non distinguables) → bonne qualité.")
print("    Un score proche de 1 indique qu'ils sont séparables → mauvaise qualité.")

sil_scores = {}
for name, scaled in [('CTGAN', ctgan_scaled), ('TVAE', tvae_scaled)]:
    X_sil    = np.vstack([real_scaled, scaled])
    y_sil    = np.array([0]*n_sample + [1]*n_sample)
    sil_score = silhouette_score(X_sil, y_sil, sample_size=min(1000, len(X_sil)),
                                  random_state=RANDOM_STATE)
    sil_scores[name] = sil_score
    print(f"\n    Silhouette {name} : {sil_score:.4f}")
    if abs(sil_score) < 0.1:
        print(f"    → Excellent : réels et synthétiques {name} quasi-indistinguables")
    elif abs(sil_score) < 0.3:
        print(f"    → Bon : légère séparabilité résiduelle entre réels et {name}")
    else:
        print(f"    → Insuffisant : {name} trop séparable des données réelles")

# ── B2. PERPLEXITÉ via discriminateur ──────────────────────────────────────────
print("\n[B2] Perplexité — via modèle discriminateur (Détection synthétique)")
print("    Principe : entraîner un classifieur à distinguer réel vs synthétique.")
print("    Si l'accuracy est proche de 50% → les synthétiques sont indistinguables.")

perp_results = {}
for name, scaled in [('CTGAN', ctgan_scaled), ('TVAE', tvae_scaled)]:
    X_disc = np.vstack([real_scaled, scaled])
    y_disc = np.array([0]*n_sample + [1]*n_sample)
    X_d_tr, X_d_te, y_d_tr, y_d_te = train_test_split(
        X_disc, y_disc, test_size=0.3, random_state=RANDOM_STATE, stratify=y_disc)
    disc = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    disc.fit(X_d_tr, y_d_tr)
    acc_disc = disc.score(X_d_te, y_d_te)
    probs    = np.clip(disc.predict_proba(X_d_te), 1e-10, 1)
    entropy  = -np.mean(np.sum(probs * np.log2(probs), axis=1))
    perp     = 2 ** entropy
    perp_results[name] = {'accuracy': acc_disc, 'perplexite': perp}
    print(f"\n    {name} : accuracy={acc_disc:.4f}, perplexité={perp:.4f}")
    if acc_disc < 0.6:
        print(f"      → Excellent : indistinguables")
    elif acc_disc < 0.75:
        print(f"      → Acceptable : légère différence détectable")
    else:
        print(f"      → Insuffisant : trop différents des réels")

# ── B3. TEST KS ────────────────────────────────────────────────────────────────
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
print("\n    → Sauvegardé : ks_test_results.csv")

# ── B4. TSTR ───────────────────────────────────────────────────────────────────
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

# ── B5. PCA + Figure synthèse non supervisée ───────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Évaluation non supervisée — Qualité des données synthétiques",
             fontsize=13, fontweight='bold', color=COLORS['neutre'])
all_data   = np.vstack([real_scaled, ctgan_scaled, tvae_scaled])
pca        = PCA(n_components=2, random_state=RANDOM_STATE)
all_pca    = pca.fit_transform(all_data)
axes[0,0].scatter(all_pca[:n_sample,0], all_pca[:n_sample,1],
                   c=COLORS['sinistre'], alpha=0.4, s=15, label='Réels')
axes[0,0].scatter(all_pca[n_sample:2*n_sample,0], all_pca[n_sample:2*n_sample,1],
                   c=COLORS['ctgan'], alpha=0.4, s=15, label='CTGAN')
axes[0,0].scatter(all_pca[2*n_sample:,0], all_pca[2*n_sample:,1],
                   c=COLORS['tvae'], alpha=0.4, s=15, label='TVAE')
axes[0,0].set_title(f"PCA — Réel vs Synthétiques\n(PC1={pca.explained_variance_ratio_[0]*100:.1f}%, "
                    f"PC2={pca.explained_variance_ratio_[1]*100:.1f}%)", fontweight='bold')
axes[0,0].legend()
df_ks_p = df_ks.dropna()
x_pos = np.arange(len(df_ks_p)); w = 0.35
axes[0,1].bar(x_pos-w/2, df_ks_p['ks_ctgan'], w, label='CTGAN', color=COLORS['ctgan'], alpha=0.8)
axes[0,1].bar(x_pos+w/2, df_ks_p['ks_tvae'],  w, label='TVAE',  color=COLORS['tvae'],  alpha=0.8)
axes[0,1].axhline(y=0.1, color='gray', linestyle='--', linewidth=1.5, label='Seuil 0.1')
axes[0,1].set_xticks(x_pos)
axes[0,1].set_xticklabels(df_ks_p['variable'], rotation=30, ha='right', fontsize=8)
axes[0,1].set_ylabel("Statistique KS (↓ mieux)")
axes[0,1].set_title("Test KS — Fidélité des distributions", fontweight='bold')
axes[0,1].legend()
names_p  = ['CTGAN', 'TVAE']
sil_vals = [abs(sil_scores['CTGAN']), abs(sil_scores['TVAE'])]
bars = axes[1,0].bar(names_p, sil_vals, color=[COLORS['ctgan'], COLORS['tvae']],
                      alpha=0.8, edgecolor='white')
axes[1,0].axhline(y=0.1, color='gray', linestyle='--', linewidth=1.5, label='Seuil 0.1 (bon)')
axes[1,0].set_ylabel("Score de Silhouette |s| (↓ mieux)")
axes[1,0].set_title("Score de Silhouette\n(0 = indistinguable des réels)", fontweight='bold')
axes[1,0].legend()
for bar, val in zip(bars, sil_vals):
    axes[1,0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
                   f'{val:.4f}', ha='center', fontweight='bold')
disc_acc = [perp_results.get('CTGAN',{}).get('accuracy', 0),
            perp_results.get('TVAE',{}).get('accuracy', 0)]
bars2 = axes[1,1].bar(names_p, disc_acc, color=[COLORS['ctgan'], COLORS['tvae']],
                       alpha=0.8, edgecolor='white')
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

# ── B6. K-MEANS CLUSTERING comparé réel vs synthétique (AJOUT) ─────────────────
print("\n[B6] (AJOUT) K-means clustering — comparaison structure réel vs synthétique")
print("    Principe : si les sinistres synthétiques sont fidèles, ils doivent")
print("    former des clusters de structure similaire aux sinistres réels.")
print("    Métriques : silhouette intra-cluster + Davies-Bouldin + Calinski-Harabasz")

k_clusters = 3
kmeans_report = []
for name, data_scaled in [('Réel', real_scaled), ('CTGAN', ctgan_scaled), ('TVAE', tvae_scaled)]:
    km = KMeans(n_clusters=k_clusters, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(data_scaled)
    sil = silhouette_score(data_scaled, labels)
    db  = davies_bouldin_score(data_scaled, labels)
    ch  = calinski_harabasz_score(data_scaled, labels)
    sizes = pd.Series(labels).value_counts().sort_index().tolist()
    kmeans_report.append({
        'Dataset': name, 'Silhouette': sil, 'Davies-Bouldin': db,
        'Calinski-Harabasz': ch, 'Tailles clusters': sizes,
    })
    print(f"    {name:<6s} : sil={sil:.4f}  DB={db:.4f}  CH={ch:.1f}  tailles={sizes}")

df_kmeans = pd.DataFrame(kmeans_report)
df_kmeans.to_csv('../outputs/kmeans_comparison.csv', index=False)
print("    → Sauvegardé : kmeans_comparison.csv")
print("\n    Interprétation : si Silhouette/DB/CH du synthé ≈ ceux du réel,")
print("    alors la structure latente est bien préservée par le générateur.")

# Visualisation : clusters K-means superposés sur PCA
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("K-means clustering (k=3) — Structure réel vs synthétique",
             fontsize=13, fontweight='bold', color=COLORS['neutre'])
for ax, (name, data_scaled, color_base) in zip(axes,
        [('Réel', real_scaled, COLORS['sinistre']),
         ('CTGAN', ctgan_scaled, COLORS['ctgan']),
         ('TVAE', tvae_scaled, COLORS['tvae'])]):
    km = KMeans(n_clusters=k_clusters, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(data_scaled)
    pca_local = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(data_scaled)
    palette = sns.color_palette("Set2", k_clusters)
    for k in range(k_clusters):
        mask = labels == k
        ax.scatter(pca_local[mask,0], pca_local[mask,1], c=[palette[k]],
                    s=25, alpha=0.7, label=f'Cluster {k} (n={mask.sum()})',
                    edgecolors='white', linewidths=0.4)
    sil_local = silhouette_score(data_scaled, labels)
    ax.set_title(f"{name} — Silhouette={sil_local:.3f}", fontweight='bold')
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.legend(fontsize=8, loc='best')
plt.tight_layout()
plt.savefig('../outputs/figures/15_kmeans_clustering.png', dpi=150, bbox_inches='tight')
plt.close()
print("    → Sauvegardé : 15_kmeans_clustering.png")

# ── B7. t-SNE — projection non-linéaire (AJOUT) ────────────────────────────────
print("\n[B7] (AJOUT) t-SNE — projection non-linéaire 2D")
print("    Complément du PCA : capture les structures non-linéaires.")
print("    Calcul long (~30 sec pour 1500 points)...")

all_data_tsne = np.vstack([real_scaled, ctgan_scaled, tvae_scaled])
tsne = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=30,
            init='pca', learning_rate='auto', n_iter=1000)
all_tsne = tsne.fit_transform(all_data_tsne)

fig, ax = plt.subplots(figsize=(10, 7))
ax.scatter(all_tsne[:n_sample,0], all_tsne[:n_sample,1],
            c=COLORS['sinistre'], alpha=0.5, s=20, label='Réels (sinistres)',
            edgecolors='white', linewidths=0.3)
ax.scatter(all_tsne[n_sample:2*n_sample,0], all_tsne[n_sample:2*n_sample,1],
            c=COLORS['ctgan'], alpha=0.5, s=20, label='Synthétique CTGAN',
            edgecolors='white', linewidths=0.3)
ax.scatter(all_tsne[2*n_sample:,0], all_tsne[2*n_sample:,1],
            c=COLORS['tvae'], alpha=0.5, s=20, label='Synthétique TVAE',
            edgecolors='white', linewidths=0.3)
ax.set_title("t-SNE — Projection non-linéaire 2D des sinistres\n"
             "Réels vs CTGAN vs TVAE", fontweight='bold', color=COLORS['neutre'])
ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
ax.legend(loc='best', fontsize=10)
plt.tight_layout()
plt.savefig('../outputs/figures/16_tsne_projection.png', dpi=150, bbox_inches='tight')
plt.close()
print("    → Sauvegardé : 16_tsne_projection.png")
print("    Interprétation : si les 3 nuages se superposent → synthétiques fidèles")
print("    Si CTGAN/TVAE forment des îlots séparés → faible fidélité distributionnelle")

# ══════════════════════════════════════════════════════════════════════════════
# C. ANALYSE DE SENSIBILITÉ
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("  C. ANALYSE DE SENSIBILITÉ")
print(f"{'='*70}")

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
ax.plot(df_thresh['threshold'], df_thresh['recall'],   label='Recall',    color=COLORS['sinistre'],     linewidth=2.5)
ax.plot(df_thresh['threshold'], df_thresh['precision'],label='Precision', color=COLORS['non_sinistre'], linewidth=2.5)
ax.axvline(x=best_thresh, color='gray', linestyle='--', linewidth=1.5,
            label=f'Seuil optimal ({best_thresh:.2f})')
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
# D. LLM-AS-A-JUDGE (AJOUT) — Évaluation qualitative des descriptions LLM
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("  D. (AJOUT) LLM-AS-A-JUDGE — Qualité des descriptions générées")
print(f"{'='*70}")
print("\n    Évaluation des descriptions générées par phi3.5 en Étape 3.")
print("    Sans appel à Ollama (offline), on utilise des heuristiques métriques :")
print("    • Longueur moyenne (mots)")
print("    • Diversité lexicale (TTR = unique_tokens / total_tokens)")
print("    • Présence de termes métier attendus (couverture vocabulaire actuarial)")

llm_path = '../outputs/llm/sinistres_avec_descriptions.csv'
if os.path.exists(llm_path):
    df_llm = pd.read_csv(llm_path)
    # Détecter la colonne de description (la plus longue en moyenne)
    desc_col = None
    for c in df_llm.columns:
        if df_llm[c].dtype == object:
            avg_len = df_llm[c].astype(str).str.len().mean()
            if avg_len > 50:
                desc_col = c
                break
    if desc_col is None:
        print("    ⚠️  Aucune colonne de description détectée.")
    else:
        print(f"    Colonne descriptions détectée : '{desc_col}' ({len(df_llm)} entrées)\n")
        descriptions = df_llm[desc_col].dropna().astype(str).tolist()

        # Métrique 1 : longueur moyenne
        lengths = [len(d.split()) for d in descriptions]
        len_mean = np.mean(lengths); len_std = np.std(lengths)

        # Métrique 2 : diversité lexicale (Type-Token Ratio)
        all_tokens = []
        for d in descriptions[:500]:
            all_tokens.extend(d.lower().split())
        ttr = len(set(all_tokens)) / max(1, len(all_tokens))

        # Métrique 3 : couverture vocabulaire actuariel
        vocab_actuarial = {'sinistre','accident','collision','dommage','assurance',
                            'véhicule','conducteur','prime','garantie','tiers',
                            'responsabilité','réparation','franchise','déclaration',
                            'expert','indemnisation','contrat','police'}
        coverage = []
        for d in descriptions[:500]:
            tokens = set(d.lower().split())
            cov = len(tokens & vocab_actuarial) / max(1, len(vocab_actuarial))
            coverage.append(cov)
        cov_mean = np.mean(coverage)

        # Score qualité global (0-10)
        score_long  = min(10, len_mean / 5)        # 50 mots = 10/10
        score_div   = min(10, ttr * 50)            # TTR=0.2 → 10/10
        score_cov   = cov_mean * 10                # 100% couverture → 10/10
        score_total = (score_long + score_div + score_cov) / 3

        print(f"    Métriques de qualité LLM-as-a-Judge (heuristiques) :")
        print(f"    -----------------------------------------------------")
        print(f"    Longueur moyenne          : {len_mean:.1f} ± {len_std:.1f} mots   → score {score_long:.1f}/10")
        print(f"    Diversité lexicale (TTR)  : {ttr:.4f}                  → score {score_div:.1f}/10")
        print(f"    Couverture vocab actuarial: {cov_mean*100:.1f}%               → score {score_cov:.1f}/10")
        print(f"    -----------------------------------------------------")
        print(f"    SCORE GLOBAL              : {score_total:.2f}/10")

        # Visualisation
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle("LLM-as-a-Judge — Qualité des descriptions générées par phi3.5",
                     fontsize=13, fontweight='bold', color=COLORS['neutre'])
        axes[0].hist(lengths, bins=30, color=COLORS['neutre'], alpha=0.8, edgecolor='white')
        axes[0].axvline(x=len_mean, color=COLORS['sinistre'], linestyle='--', lw=2,
                        label=f'Moy = {len_mean:.1f} mots')
        axes[0].set_xlabel("Nombre de mots"); axes[0].set_ylabel("Fréquence")
        axes[0].set_title("Distribution des longueurs", fontweight='bold')
        axes[0].legend()

        scores_dim = [score_long, score_div, score_cov]
        labels_dim = ['Longueur', 'Diversité', 'Vocab. actuarial']
        bars = axes[1].bar(labels_dim, scores_dim,
                            color=[COLORS['neutre'], COLORS['ctgan'], COLORS['tvae']],
                            alpha=0.8, edgecolor='white')
        axes[1].axhline(y=score_total, color=COLORS['sinistre'], linestyle='--', lw=2,
                        label=f'Score global = {score_total:.2f}/10')
        axes[1].set_ylim(0, 10); axes[1].set_ylabel("Score (0-10)")
        axes[1].set_title("Scores par dimension", fontweight='bold')
        axes[1].legend()
        for bar, val in zip(bars, scores_dim):
            axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
                        f'{val:.1f}', ha='center', fontweight='bold')

        axes[2].hist(coverage, bins=20, color=COLORS['vert'], alpha=0.8, edgecolor='white')
        axes[2].axvline(x=cov_mean, color=COLORS['sinistre'], linestyle='--', lw=2,
                        label=f'Moy = {cov_mean*100:.1f}%')
        axes[2].set_xlabel("Couverture vocab actuarial")
        axes[2].set_ylabel("Fréquence")
        axes[2].set_title("Distribution couverture", fontweight='bold')
        axes[2].legend()

        plt.tight_layout()
        plt.savefig('../outputs/figures/17_llm_as_judge.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    → Sauvegardé : 17_llm_as_judge.png")

        df_llm_scores = pd.DataFrame([{
            'longueur_moy_mots': len_mean, 'longueur_std': len_std,
            'diversite_lexicale_TTR': ttr,
            'couverture_vocab_actuarial': cov_mean,
            'score_longueur_sur_10': score_long,
            'score_diversite_sur_10': score_div,
            'score_couverture_sur_10': score_cov,
            'score_global_sur_10': score_total,
        }])
        df_llm_scores.to_csv('../outputs/llm_as_judge_scores.csv', index=False)
        print(f"    → Sauvegardé : llm_as_judge_scores.csv")
else:
    print(f"    ⚠️  Fichier {llm_path} introuvable — section LLM-Judge sautée.")
    score_total = None

# ══════════════════════════════════════════════════════════════════════════════
# E. CALIBRATION DES PROBABILITÉS (AJOUT)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("  E. (AJOUT) CALIBRATION DES PROBABILITÉS")
print(f"{'='*70}")
print("\n    Crucial en actuariat : si on tarifie selon la probabilité de sinistre,")
print("    une probabilité de 0.3 doit signifier que 30% des polices similaires")
print("    auront un sinistre. Mesure : Brier score + Reliability Diagram.")

print(f"\n    {'Modèle':<25} {'Brier (↓ mieux)':>17}")
print("    " + "-"*43)
fig, ax = plt.subplots(figsize=(8, 7))
ax.plot([0, 1], [0, 1], '--', color='gray', lw=1.5, label='Calibration parfaite')

brier_results = []
for (name, path), color in zip(model_files.items(), model_colors):
    if not os.path.exists(path):
        continue
    m = pickle.load(open(path, 'rb'))
    y_prob = m.predict_proba(X_test)[:, 1]
    brier  = brier_score_loss(y_test, y_prob)
    brier_results.append({'model': name, 'brier_score': brier})
    print(f"    {name:<25} {brier:>17.4f}")

    # Reliability diagram
    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10, strategy='quantile')
    ax.plot(prob_pred, prob_true, 'o-', color=color, lw=2, markersize=7,
            label=f"{name} (Brier={brier:.4f})")

ax.set_xlabel("Probabilité prédite moyenne")
ax.set_ylabel("Fraction réelle de sinistres")
ax.set_title("Reliability Diagram — Calibration des probabilités\n"
             "(Plus la courbe est proche de la diagonale, mieux le modèle est calibré)",
             fontweight='bold', color=COLORS['neutre'])
ax.legend(loc='upper left', fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../outputs/figures/18_calibration_reliability.png', dpi=150, bbox_inches='tight')
plt.close()
print("    → Sauvegardé : 18_calibration_reliability.png")
print("\n    Brier score interprétation :")
print("    • Brier ~0.05  : excellent (calibration parfaite ~0.0)")
print("    • Brier ~0.10  : bon")
print("    • Brier ~0.25  : équivalent à du hasard pour un dataset équilibré")
print("    En actuariat, un Brier élevé = primes mal calibrées = pertes financières.")

pd.DataFrame(brier_results).to_csv('../outputs/calibration_brier.csv', index=False)
print("    → Sauvegardé : calibration_brier.csv")

# ══════════════════════════════════════════════════════════════════════════════
# RÉSUMÉ FINAL
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("  RÉSUMÉ ÉTAPE 4 (VERSION AMÉLIORÉE)")
print(f"{'='*70}")
print(f"\n  A. Évaluation supervisée")
print(f"     Meilleur AUC    : {df_sup['auc'].max():.4f} ({df_sup.loc[df_sup['auc'].idxmax(),'model']})")
print(f"     Meilleur F1     : {df_sup['f1'].max():.4f} ({df_sup.loc[df_sup['f1'].idxmax(),'model']})")
print(f"     Meilleur Recall : {df_sup['recall'].max():.4f} ({df_sup.loc[df_sup['recall'].idxmax(),'model']})")

print(f"\n  B. Évaluation non supervisée")
for name in ['CTGAN','TVAE']:
    if name in perp_results:
        print(f"     Discriminateur {name} : accuracy={perp_results[name]['accuracy']:.4f}, "
              f"perplexité={perp_results[name]['perplexite']:.4f}")
for name, res in tstr_results.items():
    print(f"     TSTR {name}         : AUC={res['auc']:.4f}")
print(f"     K-means structure  : silhouettes Réel/CTGAN/TVAE = "
      f"{df_kmeans['Silhouette'].values[0]:.3f}/"
      f"{df_kmeans['Silhouette'].values[1]:.3f}/"
      f"{df_kmeans['Silhouette'].values[2]:.3f}")

print(f"\n  C. Analyse de sensibilité")
print(f"     Seuil optimal (F1) : {best_thresh:.2f} → F1={df_thresh['f1'].max():.4f}")
print(f"     Modèle réduit      : AUC={auc_red:.4f} (perte={auc_base-auc_red:.4f})")

if score_total is not None:
    print(f"\n  D. LLM-as-a-Judge")
    print(f"     Score global descriptions : {score_total:.2f}/10")

print(f"\n  E. Calibration probabiliste")
if len(brier_results) > 0:
    best_brier = min(brier_results, key=lambda x: x['brier_score'])
    print(f"     Meilleur Brier   : {best_brier['brier_score']:.4f} ({best_brier['model']})")

print(f"\n  Figures produites : 10 à 18 dans ../outputs/figures/")
print(f"  CSV produits      : metriques_supervisees, ks_test_results, kmeans_comparison,")
print(f"                      sensibilite_volume, llm_as_judge_scores, calibration_brier")
print(f"\n✓ Étape 4 terminée (version améliorée).")
