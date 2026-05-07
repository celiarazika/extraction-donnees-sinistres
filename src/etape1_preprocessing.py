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

# ── 1. Chargement ──────────────────────────────────────────────────────────────
df_raw = pd.read_csv('../data/Insurance claims data.csv', sep=',')
print(f"[1] Données brutes : {df_raw.shape[0]} lignes, {df_raw.shape[1]} colonnes")
print(f"    Valeurs manquantes : {df_raw.isnull().sum().sum()}")

# ── 2. Suppression identifiant ─────────────────────────────────────────────────
df = df_raw.drop(columns=['policy_id'])
print(f"[2] Suppression de policy_id → {df.shape[1]} colonnes restantes")

# ── 3. Parsing max_torque et max_power ─────────────────────────────────────────
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
print(f"[3] Parsing max_torque/max_power → 4 colonnes numériques extraites")

# ── 4. Encodage des colonnes binaires Yes/No ───────────────────────────────────
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
print(f"[4] Encodage binaire Yes/No sur {len(binary_cols)} colonnes")

# ── 5. Encodage des colonnes catégorielles nominales ──────────────────────────
nominal_cols = ['fuel_type', 'transmission_type', 'rear_brakes_type',
                'steering_type', 'segment', 'engine_type', 'model',
                'region_code']
df_encoded = pd.get_dummies(df, columns=nominal_cols, drop_first=False)
print(f"[5] One-hot encoding sur {len(nominal_cols)} colonnes catégorielles")
print(f"    Dimensions après encoding : {df_encoded.shape}")

# ── 6. Vérification déséquilibre ───────────────────────────────────────────────
n_claims = df_encoded['claim_status'].sum()
n_total  = len(df_encoded)
ratio    = n_claims / n_total * 100
print(f"\n[6] Déséquilibre des classes :")
print(f"    Sinistres (1)     : {int(n_claims):>6} ({ratio:.1f}%)")
print(f"    Non-sinistres (0) : {n_total - int(n_claims):>6} ({100-ratio:.1f}%)")
print(f"    Ratio d'imbalance : 1 pour {int((n_total - n_claims) / n_claims)}")

# ── 7. Séparation features / cible ────────────────────────────────────────────
X = df_encoded.drop(columns=['claim_status'])
y = df_encoded['claim_status']

# ── 8. Normalisation des colonnes numériques continues ────────────────────────
numeric_cols = [
    'subscription_length', 'vehicle_age', 'customer_age', 'region_density',
    'displacement', 'cylinder', 'turning_radius', 'length', 'width',
    'gross_weight', 'torque_nm', 'torque_rpm', 'power_bhp', 'power_rpm', 'airbags'
]
scaler = StandardScaler()
X_scaled = X.copy()
X_scaled[numeric_cols] = scaler.fit_transform(X[numeric_cols])
print(f"\n[7] Normalisation (StandardScaler) sur {len(numeric_cols)} colonnes numériques")

# ── 9. Sauvegarde ──────────────────────────────────────────────────────────────
os.makedirs('../outputs', exist_ok=True)
X_scaled['claim_status'] = y.values
X_scaled.to_csv('../outputs/data_preprocessed.csv', index=False)
df_encoded.to_csv('../outputs/data_encoded.csv', index=False)

print(f"\n[8] Fichiers sauvegardés dans outputs/")
print(f"    data_preprocessed.csv ({X_scaled.shape[0]} x {X_scaled.shape[1]})")
print(f"    data_encoded.csv       ({df_encoded.shape[0]} x {df_encoded.shape[1]})")
print(f"\n✓ Étape 1 terminée.")
