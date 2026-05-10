"""
Étape 3 – Modélisation
Projet : Génération de données synthétiques de sinistres pour la tarification
Dataset : Insurance Claims Data (58 592 polices)

Approche retenue : ML / Deep Learning Génératif + LLM
  - CTGAN (Conditional Tabular GAN) — génération de sinistres synthétiques
  - TVAE (Tabular Variational Autoencoder) — alternative au GAN
  - XGBoost — classifieur downstream standard actuariel
  - SMOTE — baseline d'augmentation classique
  - LLM Ollama/phi3.5 — génération de descriptions textuelles de sinistres

Justification du choix ML/DL + LLM :
  Le projet mobilise les deux approches autorisées par les consignes :
  1. ML/DL génératif (CTGAN/TVAE) pour la génération de données tabulaires
  2. Architecture LLM locale (Ollama/phi3.5) pour l'enrichissement textuel
  Les LLMs classiques (GPT, Claude) sont inadaptés à la génération tabulaire
  mais parfaitement adaptés à la rédaction automatique de dossiers sinistres.

Installation des dépendances :
  pip install sdv xgboost imbalanced-learn openai
  + Ollama installé localement : https://ollama.com
  + Modèle phi3.5 : ollama pull phi3.5
"""

import pandas as pd
import numpy as np
import os
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, f1_score, recall_score,
                             precision_score, confusion_matrix)
import xgboost as xgb

try:
    from sdv.single_table import CTGANSynthesizer, TVAESynthesizer
    from sdv.metadata import SingleTableMetadata
    SDV_AVAILABLE = True
except ImportError:
    print("⚠️  SDV non installé : pip install sdv")
    SDV_AVAILABLE = False

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    print("⚠️  imbalanced-learn non installé : pip install imbalanced-learn")
    SMOTE_AVAILABLE = False

os.makedirs('../outputs/models', exist_ok=True)
os.makedirs('../outputs/synthetic', exist_ok=True)
os.makedirs('../outputs/llm', exist_ok=True)

# ── Configuration ──────────────────────────────────────────────────────────────
N_SYNTHETIC  = 5000
EPOCHS_CTGAN = 300
EPOCHS_TVAE  = 300
RANDOM_STATE = 42
TEST_SIZE    = 0.2
N_LLM        = 10  # Nombre de descriptions LLM à générer

# ══════════════════════════════════════════════════════════════════════════════
# JUSTIFICATION DE L'APPROCHE RETENUE
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("  ÉTAPE 3 — MODÉLISATION")
print("=" * 65)
print("""
JUSTIFICATION DE L'APPROCHE RETENUE
─────────────────────────────────────────
Le projet mobilise les DEUX approches autorisées par les consignes :

[1] ML/DL GÉNÉRATIF (CTGAN + TVAE + XGBoost)
  Pourquoi : les données sont tabulaires (58 592 × 94 colonnes numériques)
  CTGAN/TVAE sont les architectures state-of-the-art (MIT) pour ce cas.
  XGBoost est le standard actuariel pour la classification binaire.

[2] ARCHITECTURE LLM (Ollama/phi3.5 — modèle local gratuit)
  Pourquoi : enrichir les sinistres synthétiques avec des descriptions
  textuelles professionnelles, comme un expert en assurance le ferait.
  Ollama permet d'utiliser un LLM en local sans coût d'API.

Pipeline complet :
  Données réelles → CTGAN/TVAE → sinistres synthétiques (chiffres)
                                          ↓
                              Ollama/phi3.5 → descriptions textuelles
                                          ↓
                              Dossier sinistre complet
─────────────────────────────────────────
""")

# ══════════════════════════════════════════════════════════════════════════════
# 1. CHARGEMENT DES DONNÉES
# ══════════════════════════════════════════════════════════════════════════════
df_encoded = pd.read_csv('../outputs/data_encoded.csv')
df_pre     = pd.read_csv('../outputs/data_preprocessed.csv')
df_raw     = pd.read_csv('../data/Insurance claims data.csv', sep=',')

print(f"[1] Données chargées :")
print(f"    data_encoded.csv      : {df_encoded.shape}")
print(f"    data_preprocessed.csv : {df_pre.shape}")

X = df_pre.drop(columns=['claim_status'])
y = df_pre['claim_status']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print(f"\n[2] Split train/test (stratifié, test={TEST_SIZE}) :")
print(f"    Train : {X_train.shape[0]:,} — {int(y_train.sum()):,} sinistres ({y_train.mean()*100:.1f}%)")
print(f"    Test  : {X_test.shape[0]:,}  — {int(y_test.sum()):,} sinistres ({y_test.mean()*100:.1f}%)")

# ── Fonction d'évaluation ──────────────────────────────────────────────────────
def evaluate_model(model, X_test, y_test, model_name):
    y_pred      = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    auc  = roc_auc_score(y_test, y_pred_prob)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    prec = precision_score(y_test, y_pred, zero_division=0)
    cm   = confusion_matrix(y_test, y_pred)
    print(f"\n  ── {model_name} ──")
    print(f"    AUC-ROC   : {auc:.4f}")
    print(f"    F1-score  : {f1:.4f}")
    print(f"    Recall    : {rec:.4f}")
    print(f"    Precision : {prec:.4f}")
    print(f"    Confusion :")
    print(f"      TN={cm[0,0]:>5}  FP={cm[0,1]:>5}")
    print(f"      FN={cm[1,0]:>5}  TP={cm[1,1]:>5}")
    return {'model': model_name, 'auc': auc, 'f1': f1, 'recall': rec, 'precision': prec}

results = []

# ══════════════════════════════════════════════════════════════════════════════
# 2. BASELINE — XGBoost sur données brutes
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("  [3] BASELINE — XGBoost (données brutes, sans augmentation)")
print(f"{'='*65}")

scale_pos_weight = int((y_train == 0).sum() / (y_train == 1).sum())
print(f"  scale_pos_weight = {scale_pos_weight} (compense le déséquilibre 1/14)")

xgb_baseline = xgb.XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=RANDOM_STATE, eval_metric='auc', verbosity=0
)
xgb_baseline.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
res_baseline = evaluate_model(xgb_baseline, X_test, y_test, "XGBoost Baseline")
results.append(res_baseline)
pickle.dump(xgb_baseline, open('../outputs/models/xgb_baseline.pkl', 'wb'))
print("  → Sauvegardé : outputs/models/xgb_baseline.pkl")

# ══════════════════════════════════════════════════════════════════════════════
# 3. GÉNÉRATION CTGAN
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("  [4] GÉNÉRATION CTGAN")
print(f"{'='*65}")
print("  Architecture : Conditional Tabular GAN (Xu et al., MIT, 2019)")
print(f"  Paramètres   : epochs={EPOCHS_CTGAN}, n_synthetic={N_SYNTHETIC}")

CTGAN_AVAILABLE = False
if SDV_AVAILABLE:
    df_train_full = X_train.copy()
    df_train_full['claim_status'] = y_train.values
    df_sinistres  = df_train_full[df_train_full['claim_status'] == 1].drop(columns=['claim_status'])
    print(f"\n  Entraînement sur {len(df_sinistres):,} sinistres réels...")

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df_sinistres)

    ctgan = CTGANSynthesizer(metadata, epochs=EPOCHS_CTGAN, verbose=True)
    ctgan.fit(df_sinistres)
    print("  ✓ CTGAN entraîné")

    synthetic_ctgan = ctgan.sample(num_rows=N_SYNTHETIC)
    synthetic_ctgan['claim_status'] = 1
    synthetic_ctgan.to_csv('../outputs/synthetic/synthetic_ctgan.csv', index=False)
    ctgan.save('../outputs/models/ctgan_model.pkl')
    print(f"  ✓ {N_SYNTHETIC:,} sinistres synthétiques CTGAN générés")
    CTGAN_AVAILABLE = True
else:
    print("  ⚠️  SDV non disponible — pip install sdv")

# ══════════════════════════════════════════════════════════════════════════════
# 4. GÉNÉRATION TVAE
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("  [5] GÉNÉRATION TVAE")
print(f"{'='*65}")
print("  Architecture : Tabular Variational Autoencoder (Xu et al., MIT, 2019)")
print(f"  Paramètres   : epochs={EPOCHS_TVAE}, n_synthetic={N_SYNTHETIC}")

TVAE_AVAILABLE = False
if SDV_AVAILABLE:
    tvae = TVAESynthesizer(metadata, epochs=EPOCHS_TVAE, verbose=True)
    tvae.fit(df_sinistres)
    print("  ✓ TVAE entraîné")

    synthetic_tvae = tvae.sample(num_rows=N_SYNTHETIC)
    synthetic_tvae['claim_status'] = 1
    synthetic_tvae.to_csv('../outputs/synthetic/synthetic_tvae.csv', index=False)
    tvae.save('../outputs/models/tvae_model.pkl')
    print(f"  ✓ {N_SYNTHETIC:,} sinistres synthétiques TVAE générés")
    TVAE_AVAILABLE = True
else:
    print("  ⚠️  SDV non disponible — pip install sdv")

# ══════════════════════════════════════════════════════════════════════════════
# 5. CLASSIFIEUR AUGMENTÉ — XGBoost + CTGAN
# ══════════════════════════════════════════════════════════════════════════════
if CTGAN_AVAILABLE:
    print(f"\n{'='*65}")
    print("  [6] CLASSIFIEUR AUGMENTÉ — XGBoost + CTGAN")
    print(f"{'='*65}")

    syn_X = synthetic_ctgan.drop(columns=['claim_status'], errors='ignore')
    syn_X = syn_X.reindex(columns=X_train.columns, fill_value=0)
    X_aug = pd.concat([X_train, syn_X], ignore_index=True)
    y_aug = pd.concat([y_train, pd.Series([1]*N_SYNTHETIC)], ignore_index=True)
    spw   = max(1, int((y_aug==0).sum() / (y_aug==1).sum()))

    print(f"  Train augmenté : {len(X_aug):,} polices — {int(y_aug.sum()):,} sinistres ({y_aug.mean()*100:.1f}%)")
    print(f"  scale_pos_weight ajusté : {spw}")

    xgb_ctgan = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw,
        random_state=RANDOM_STATE, eval_metric='auc', verbosity=0
    )
    xgb_ctgan.fit(X_aug, y_aug, eval_set=[(X_test, y_test)], verbose=False)
    res_ctgan = evaluate_model(xgb_ctgan, X_test, y_test, "XGBoost + CTGAN")
    results.append(res_ctgan)
    pickle.dump(xgb_ctgan, open('../outputs/models/xgb_ctgan.pkl', 'wb'))
    print("  → Sauvegardé : outputs/models/xgb_ctgan.pkl")

# ══════════════════════════════════════════════════════════════════════════════
# 6. CLASSIFIEUR AUGMENTÉ — XGBoost + TVAE
# ══════════════════════════════════════════════════════════════════════════════
if TVAE_AVAILABLE:
    print(f"\n{'='*65}")
    print("  [7] CLASSIFIEUR AUGMENTÉ — XGBoost + TVAE")
    print(f"{'='*65}")

    syn_X = synthetic_tvae.drop(columns=['claim_status'], errors='ignore')
    syn_X = syn_X.reindex(columns=X_train.columns, fill_value=0)
    X_aug = pd.concat([X_train, syn_X], ignore_index=True)
    y_aug = pd.concat([y_train, pd.Series([1]*N_SYNTHETIC)], ignore_index=True)
    spw   = max(1, int((y_aug==0).sum() / (y_aug==1).sum()))

    print(f"  Train augmenté : {len(X_aug):,} polices — {int(y_aug.sum()):,} sinistres ({y_aug.mean()*100:.1f}%)")
    print(f"  scale_pos_weight ajusté : {spw}")

    xgb_tvae = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw,
        random_state=RANDOM_STATE, eval_metric='auc', verbosity=0
    )
    xgb_tvae.fit(X_aug, y_aug, eval_set=[(X_test, y_test)], verbose=False)
    res_tvae = evaluate_model(xgb_tvae, X_test, y_test, "XGBoost + TVAE")
    results.append(res_tvae)
    pickle.dump(xgb_tvae, open('../outputs/models/xgb_tvae.pkl', 'wb'))
    print("  → Sauvegardé : outputs/models/xgb_tvae.pkl")

# ══════════════════════════════════════════════════════════════════════════════
# 7. SMOTE — Baseline d'augmentation classique
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("  [8] SMOTE — Baseline d'augmentation classique")
print(f"{'='*65}")

if SMOTE_AVAILABLE:
    smote = SMOTE(random_state=RANDOM_STATE)
    X_smote, y_smote = smote.fit_resample(X_train, y_train)
    spw = max(1, int((y_smote==0).sum() / (y_smote==1).sum()))

    print(f"  Train augmenté SMOTE : {len(X_smote):,} polices — {int(y_smote.sum()):,} sinistres ({y_smote.mean()*100:.1f}%)")
    print(f"  scale_pos_weight ajusté : {spw}")

    xgb_smote = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw,
        random_state=RANDOM_STATE, eval_metric='auc', verbosity=0
    )
    xgb_smote.fit(X_smote, y_smote, eval_set=[(X_test, y_test)], verbose=False)
    res_smote = evaluate_model(xgb_smote, X_test, y_test, "XGBoost + SMOTE")
    results.append(res_smote)
    pickle.dump(xgb_smote, open('../outputs/models/xgb_smote.pkl', 'wb'))
    print("  → Sauvegardé : outputs/models/xgb_smote.pkl")
else:
    print("  ⚠️  imbalanced-learn non disponible — pip install imbalanced-learn")

# ══════════════════════════════════════════════════════════════════════════════
# 8. GÉNÉRATION LLM — Descriptions textuelles des sinistres
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("  [9] GÉNÉRATION LLM — Ollama/phi3.5")
print(f"{'='*65}")
print("  Architecture : LLM local via Ollama (OpenAI-compatible API)")
print("  Modèle       : phi3.5 (Microsoft, rapide et précis)")
print("  Objectif     : générer des descriptions textuelles professionnelles")
print("                 pour les sinistres synthétiques CTGAN")
print(f"  N descriptions : {N_LLM}")

class ClaimsLLMGenerator:
    """Génère des descriptions de sinistres via LLM local (Ollama)."""

    def __init__(self, model_name: str = "ollama"):
        self.model_name = model_name
        self.client     = None
        self._load_model()

    def _load_model(self):
        try:
            from openai import OpenAI
            self.client = OpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama"
            )
            print("  ✓ Ollama API configurée (local — gratuit)")
        except ImportError:
            print("  ⚠️  openai non installé : pip install openai")
            self.client = None

    def create_prompt(self, claim_data) -> str:
        if isinstance(claim_data, str):
            claim_details = claim_data
        else:
            claim_details = "\n".join(f"- {k}: {v}" for k, v in claim_data.items())

        return f"""Tu es un expert en sinistres d'assurance automobile. Analyse ces données et rédige une description concise et précise du dossier:

DONNÉES DU DOSSIER:
{claim_details}

Rédige une description professionnelle de 50-100 mots basée UNIQUEMENT sur les informations fournies. Pas d'inventions, pas d'hypothèses:
"""

    def generate(self, claim_data) -> str:
        if self.client is None:
            return "LLM non disponible"
        prompt = self.create_prompt(claim_data)
        try:
            response = self.client.chat.completions.create(
                model="phi3.5",
                messages=[
                    {"role": "system", "content": "You are a senior insurance claims expert with 20 years of experience."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.2,
                top_p=0.9
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Erreur génération : {e}"

    def generate_batch(self, claims_data) -> list:
        descriptions = []
        for i, claim in enumerate(claims_data):
            print(f"  Génération {i+1}/{len(claims_data)}...", end='\r')
            descriptions.append(self.generate(claim))
        print(f"\n  ✓ {len(descriptions)} descriptions générées")
        return descriptions


# Instanciation et génération
llm_generator = ClaimsLLMGenerator()

if llm_generator.client is not None and CTGAN_AVAILABLE:
    # Préparer les sinistres synthétiques pour le LLM
    # On utilise les colonnes originales du dataset brut pour un meilleur contexte
    raw_cols = ['customer_age', 'vehicle_age', 'subscription_length',
                'region_density', 'fuel_type', 'segment',
                'transmission_type', 'airbags', 'ncap_rating']

    # Utiliser les sinistres réels comme référence pour le contexte
    df_claims_raw = df_raw[df_raw['claim_status'] == 1][raw_cols].head(N_LLM)
    claims_list   = df_claims_raw.to_dict(orient='records')

    print(f"\n  Génération de {N_LLM} descriptions pour des sinistres réels...")
    descriptions = llm_generator.generate_batch(claims_list)

    # Sauvegarde
    df_llm = df_claims_raw.copy().reset_index(drop=True)
    df_llm['description_llm'] = descriptions
    df_llm.to_csv('../outputs/llm/sinistres_avec_descriptions.csv', index=False)

    print(f"\n  Exemple de description générée :")
    print(f"  {'-'*50}")
    print(f"  Données : {claims_list[0]}")
    print(f"  Description LLM :")
    print(f"  {descriptions[0]}")
    print(f"  {'-'*50}")
    print("  → Sauvegardé : outputs/llm/sinistres_avec_descriptions.csv")

elif llm_generator.client is None:
    print("\n  ⚠️  Ollama non disponible.")
    print("  Pour l'activer :")
    print("    1. Installer Ollama : https://ollama.com")
    print("    2. Télécharger le modèle : ollama pull phi3.5")
    print("    3. Lancer Ollama : ollama serve")
    print("    4. Relancer ce script")

# ══════════════════════════════════════════════════════════════════════════════
# 9. TABLEAU COMPARATIF FINAL
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("  TABLEAU COMPARATIF FINAL")
print(f"{'='*65}")

df_results = pd.DataFrame(results).sort_values('f1', ascending=False)
df_results.to_csv('../outputs/resultats_modelisation.csv', index=False)

print(f"\n  {'Modèle':<30} {'AUC':>8} {'F1':>8} {'Recall':>8} {'Precision':>8}")
print("  " + "-"*60)
for _, row in df_results.iterrows():
    print(f"  {row['model']:<30} {row['auc']:>8.4f} {row['f1']:>8.4f} "
          f"{row['recall']:>8.4f} {row['precision']:>8.4f}")

# Sauvegarde métadonnées
model_info = {
    'feature_columns': X.columns.tolist(),
    'n_features'     : len(X.columns),
    'test_size'      : TEST_SIZE,
    'n_synthetic'    : N_SYNTHETIC,
    'random_state'   : RANDOM_STATE
}
with open('../outputs/models/model_info.json', 'w') as f:
    json.dump(model_info, f, indent=2)

print(f"\n  → outputs/resultats_modelisation.csv")
print(f"  → outputs/models/model_info.json")
print(f"  → outputs/llm/sinistres_avec_descriptions.csv")
print(f"\n✓ Étape 3 terminée.")