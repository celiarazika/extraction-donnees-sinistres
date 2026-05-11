"""
Étape 1 – Data Acquisition & Prétraitement
Projet : Génération de données synthétiques de sinistres pour la tarification
Dataset : Insurance Claims Data (58 592 polices, 41 variables)
"""

import pandas as pd
import numpy as np
import re
import os
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
# 1. ACQUISITION — Chargement des données brutes
# ══════════════════════════════════════════════════════════════════════════════
df_raw = pd.read_csv('../data/Insurance claims data.csv', sep=',')
print(f"[1] Données brutes : {df_raw.shape[0]} lignes, {df_raw.shape[1]} colonnes")
print(f"    Valeurs manquantes : {df_raw.isnull().sum().sum()}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. CONTRÔLE QUALITÉ
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2] Contrôle qualité :")

# 2a. Doublons
n_doublons = df_raw.duplicated().sum()
print(f"    Doublons : {n_doublons}")
if n_doublons > 0:
    df_raw = df_raw.drop_duplicates()
    print(f"    → {n_doublons} doublon(s) supprimé(s)")

# 2b. Valeurs manquantes par colonne
n_missing = df_raw.isnull().sum()
cols_missing = n_missing[n_missing > 0]
if len(cols_missing) == 0:
    print(f"    Valeurs manquantes : aucune")
else:
    print(f"    Valeurs manquantes détectées :")
    print(cols_missing)

# 2c. Vérification des plages de valeurs (cohérence métier)
print(f"    Plages de valeurs :")
print(f"      customer_age        : [{df_raw['customer_age'].min()} – {df_raw['customer_age'].max()}] ans")
print(f"      vehicle_age         : [{df_raw['vehicle_age'].min()} – {df_raw['vehicle_age'].max()}] ans")
print(f"      subscription_length : [{df_raw['subscription_length'].min()} – {df_raw['subscription_length'].max()}] ans")
print(f"      region_density      : [{df_raw['region_density'].min()} – {df_raw['region_density'].max()}]")
print(f"      airbags             : [{df_raw['airbags'].min()} – {df_raw['airbags'].max()}]")
print(f"      ncap_rating         : [{df_raw['ncap_rating'].min()} – {df_raw['ncap_rating'].max()}]")

# Vérifications de cohérence métier
assert (df_raw['customer_age'] >= 18).all(), "Âge client < 18 détecté"
assert (df_raw['vehicle_age'] >= 0).all(),   "Âge véhicule négatif détecté"
assert (df_raw['airbags'] >= 0).all(),        "Nombre d'airbags négatif détecté"
assert df_raw['claim_status'].isin([0, 1]).all(), "claim_status contient des valeurs hors {0,1}"
print(f"    Vérifications de cohérence métier : OK")

# 2d. Détection des outliers via IQR
print(f"    Outliers détectés (méthode IQR) :")
numeric_check = ['customer_age', 'vehicle_age', 'region_density', 'displacement', 'gross_weight']
outlier_report = {}
for col in numeric_check:
    Q1  = df_raw[col].quantile(0.25)
    Q3  = df_raw[col].quantile(0.75)
    IQR = Q3 - Q1
    mask = (df_raw[col] < Q1 - 1.5 * IQR) | (df_raw[col] > Q3 + 1.5 * IQR)
    n_out = mask.sum()
    outlier_report[col] = n_out
    print(f"      {col:<25} : {n_out} outliers ({n_out/len(df_raw)*100:.1f}%)")

# Justification : on conserve les outliers — en assurance, les valeurs extrêmes
# (véhicules très anciens, densités très élevées) sont des cas réels à modéliser.
print(f"    → Outliers conservés (cas réels pertinents pour la modélisation actuarielle)")

# ══════════════════════════════════════════════════════════════════════════════
# 3. PRÉPARATION — Suppression de l'identifiant
# ══════════════════════════════════════════════════════════════════════════════
df = df_raw.drop(columns=['policy_id'])
print(f"\n[3] Suppression de policy_id → {df.shape[1]} colonnes restantes")

# ══════════════════════════════════════════════════════════════════════════════
# 4. TRANSFORMATION — Parsing max_torque et max_power
# ══════════════════════════════════════════════════════════════════════════════
def parse_torque(s):
    m = re.search(r'([\d.]+)Nm', str(s))
    return float(m.group(1)) if m else np.nan

def parse_torque_rpm(s):
    m = re.search(r'@([\d.]+)rpm', str(s))
    return float(m.group(1)) if m else np.nan

def parse_power(s):
    m = re.search(r'([\d.]+)bhp', str(s))
    return float(m.group(1)) if m else np.nan

def parse_power_rpm(s):
    m = re.search(r'@([\d.]+)rpm', str(s))
    return float(m.group(1)) if m else np.nan

df['torque_nm']  = df['max_torque'].apply(parse_torque)
df['torque_rpm'] = df['max_torque'].apply(parse_torque_rpm)
df['power_bhp']  = df['max_power'].apply(parse_power)
df['power_rpm']  = df['max_power'].apply(parse_power_rpm)
df = df.drop(columns=['max_torque', 'max_power'])
print(f"[4] Parsing max_torque/max_power → 4 colonnes numériques extraites")
print(f"    NaN après parsing : torque_nm={df['torque_nm'].isnull().sum()}, power_bhp={df['power_bhp'].isnull().sum()}")

# ══════════════════════════════════════════════════════════════════════════════
# 5. TRANSFORMATION — Encodage binaire Yes/No → 0/1
# ══════════════════════════════════════════════════════════════════════════════
binary_cols = [
    'is_esc', 'is_adjustable_steering', 'is_tpms', 'is_parking_sensors',
    'is_parking_camera', 'is_front_fog_lights', 'is_rear_window_wiper',
    'is_rear_window_washer', 'is_rear_window_defogger', 'is_brake_assist',
    'is_power_door_locks', 'is_central_locking', 'is_power_steering',
    'is_driver_seat_height_adjustable', 'is_day_night_rear_view_mirror',
    'is_ecw', 'is_speed_alert'
]
for col in binary_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0})
print(f"[5] Encodage binaire Yes/No sur {len(binary_cols)} colonnes")

# ══════════════════════════════════════════════════════════════════════════════
# 6. TRANSFORMATION — One-hot encoding des variables catégorielles
# ══════════════════════════════════════════════════════════════════════════════
nominal_cols = ['fuel_type', 'transmission_type', 'rear_brakes_type',
                'steering_type', 'segment', 'engine_type', 'model', 'region_code']
df_encoded = pd.get_dummies(df, columns=nominal_cols, drop_first=False)
print(f"[6] One-hot encoding sur {len(nominal_cols)} colonnes catégorielles")
print(f"    Dimensions après encoding : {df_encoded.shape}")

# ══════════════════════════════════════════════════════════════════════════════
# 7. DÉSÉQUILIBRE DES CLASSES
# ══════════════════════════════════════════════════════════════════════════════
n_claims = df_encoded['claim_status'].sum()
n_total  = len(df_encoded)
ratio    = n_claims / n_total * 100
print(f"\n[7] Déséquilibre des classes :")
print(f"    Sinistres (1)     : {int(n_claims):>6} ({ratio:.1f}%)")
print(f"    Non-sinistres (0) : {n_total - int(n_claims):>6} ({100-ratio:.1f}%)")
print(f"    Ratio d'imbalance : 1 pour {int((n_total - n_claims) / n_claims)}")

# ══════════════════════════════════════════════════════════════════════════════
# 8. TRANSFORMATION — Normalisation StandardScaler
# ══════════════════════════════════════════════════════════════════════════════
X = df_encoded.drop(columns=['claim_status'])
y = df_encoded['claim_status']

numeric_cols = [
    'subscription_length', 'vehicle_age', 'customer_age', 'region_density',
    'displacement', 'cylinder', 'turning_radius', 'length', 'width',
    'gross_weight', 'torque_nm', 'torque_rpm', 'power_bhp', 'power_rpm',
    'airbags', 'ncap_rating'
]
scaler   = StandardScaler()
X_scaled = X.copy()
X_scaled[numeric_cols] = scaler.fit_transform(X[numeric_cols])
print(f"\n[8] Normalisation (StandardScaler) sur {len(numeric_cols)} colonnes numériques")

# Contrôle post-normalisation
for col in numeric_cols[:3]:
    print(f"    {col:<25} → mean={X_scaled[col].mean():.4f}, std={X_scaled[col].std():.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# 9. SAUVEGARDE
# ══════════════════════════════════════════════════════════════════════════════
os.makedirs('../outputs', exist_ok=True)
X_scaled['claim_status'] = y.values
X_scaled.to_csv('../outputs/data_preprocessed.csv', index=False)
df_encoded.to_csv('../outputs/data_encoded.csv', index=False)

print(f"\n[9] Fichiers sauvegardés dans outputs/")
print(f"    data_preprocessed.csv ({X_scaled.shape[0]} x {X_scaled.shape[1]})")
print(f"    data_encoded.csv       ({df_encoded.shape[0]} x {df_encoded.shape[1]})")
print(f"\n✓ Étape 1 terminée.")