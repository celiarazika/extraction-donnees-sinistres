"""
Data processing module for insurance claims extraction.
Handles loading, cleaning, encoding, and normalization of data.
"""

import os
import pickle
import re
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings('ignore')


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


class DataProcessor:
    """Processes insurance claims data for machine learning."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.numeric_cols = []
        self.categorical_cols = []
        self.numeric_cols_original = []
    
    def load_data(self, filepath):
        """Load CSV data."""
        return pd.read_csv(filepath)
  
    def analyze_quality(self, df):
        """Analyze data quality: missing values, duplicates, dtypes."""
        print(f"\nDimensions: {df.shape}")
        print(f"\nTypes de données:\n{df.dtypes}")
        
        # Valeurs manquantes
        missing = df.isnull().sum()
        if missing.sum() > 0:
            missing_pct = (missing / len(df)) * 100
            missing_df = pd.DataFrame({
                'Colonne': missing.index,
                'Manquantes': missing.values,
                'Pourcentage': missing_pct.values
            }).sort_values('Pourcentage', ascending=False)
            print(f"\nValeurs manquantes:\n{missing_df[missing_df['Manquantes'] > 0]}")
        else:
            print("\nAucune valeur manquante détectée")
        
        # Doublons
        duplicates = df.duplicated().sum()
        print(f"\nDoublons: {duplicates}")
        
        return missing.sum() > 0, duplicates > 0

    def clean_data(self, df):
        """Clean data: remove duplicates and handle missing values."""
        df_clean = df.copy()
        
        print(f"\nÉtat initial: {df_clean.shape}")
        
        # Supprimer les doublons
        df_clean = df_clean.drop_duplicates()
        print(f"Après suppression des doublons: {df_clean.shape}")
        
        # Traiter les valeurs manquantes
        missing_cols = df_clean.columns[df_clean.isnull().any()].tolist()
        
        if missing_cols:
            print(f"\nTraitement des colonnes avec valeurs manquantes:")
            for col in missing_cols:
                if df_clean[col].dtype in ['float64', 'int64']:
                    df_clean[col].fillna(df_clean[col].median(), inplace=True)
                    print(f"  {col}: remplissage avec la médiane")
                else:
                    mode_val = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown'
                    df_clean[col].fillna(mode_val, inplace=True)
                    print(f"  {col}: remplissage avec le mode")
        
        print(f"\nAprès nettoyage: {df_clean.isnull().sum().sum()} valeurs manquantes restantes")
        
        return df_clean
    
    def detect_outliers(self, df):
        """Detect outliers using IQR method."""
        
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        outlier_info = {}
        
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            if len(outliers) > 0:
                outlier_info[col] = {
                    'count': len(outliers),
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound
                }
                print(f"\n{col}: {len(outliers)} valeurs aberrantes")
                print(f"  Plage normale: [{lower_bound:.2f}, {upper_bound:.2f}]")
        
        return outlier_info

    def parsing_power(self, df):
        """Extract numeric torque/power features from the raw string columns."""
        df = df.copy()

        if 'max_torque' in df.columns:
            df['torque_nm'] = df['max_torque'].apply(parse_torque)
            df['torque_rpm'] = df['max_torque'].apply(parse_torque_rpm)

        if 'max_power' in df.columns:
            df['power_bhp'] = df['max_power'].apply(parse_power)
            df['power_rpm'] = df['max_power'].apply(parse_power_rpm)

        drop_cols = [col for col in ['max_torque', 'max_power'] if col in df.columns]
        if drop_cols:
            df = df.drop(columns=drop_cols)

        print("[3] Parsing max_torque/max_power → 4 colonnes numériques extraites")
        return df

    def transform_data(self, df):
        """Encode categorical variables and normalize numeric variables."""
        df_transformed = df.copy()
        
        # Identifier les colonnes
        self.categorical_cols = df_transformed.select_dtypes(include=['object']).columns.tolist()
        self.numeric_cols = df_transformed.select_dtypes(include=['float64', 'int64']).columns.tolist()
        self.numeric_cols_original = self.numeric_cols.copy()
        
        print(f"\nVariables catégorielles: {len(self.categorical_cols)}")
        print(f"Variables numériques: {len(self.numeric_cols)}")
        
        # Encodage des variables catégorielles
        print("\nEncodage des variables catégorielles:")
        for col in self.categorical_cols:
            le = LabelEncoder()
            df_transformed[col] = le.fit_transform(df_transformed[col].astype(str))
            self.label_encoders[col] = le
            print(f"  {col}: {len(le.classes_)} catégories")
        
        # Normalisation des variables numériques
        print("\nNormalisation des variables numériques:")
        df_transformed[self.numeric_cols] = self.scaler.fit_transform(
            df_transformed[self.numeric_cols]
        )
        print(f"  {len(self.numeric_cols)} colonnes normalisées")
        
        return df_transformed

    def preprocess_claims(self, df):
        """Apply the new feature-engineering steps before the standard transform."""
        df = df.copy()

        if 'policy_id' in df.columns:
            df = df.drop(columns=['policy_id'])
            print(f"[2] Suppression de policy_id → {df.shape[1]} colonnes restantes")

        df = self.parsing_power(df)

        binary_cols = [
            'is_esc', 'is_adjustable_steering', 'is_tpms', 'is_parking_sensors',
            'is_parking_camera', 'is_front_fog_lights', 'is_rear_window_wiper',
            'is_rear_window_washer', 'is_rear_window_defogger', 'is_brake_assist',
            'is_power_door_locks', 'is_central_locking', 'is_power_steering',
            'is_driver_seat_height_adjustable', 'is_day_night_rear_view_mirror',
            'is_ecw', 'is_speed_alert'
        ]
        existing_binary_cols = [col for col in binary_cols if col in df.columns]
        for col in existing_binary_cols:
            df[col] = df[col].map({'Yes': 1, 'No': 0})
        print(f"[4] Encodage binaire Yes/No sur {len(existing_binary_cols)} colonnes")

        nominal_cols = [
            'fuel_type', 'transmission_type', 'rear_brakes_type',
            'steering_type', 'segment', 'engine_type', 'model', 'region_code'
        ]
        existing_nominal_cols = [col for col in nominal_cols if col in df.columns]
        df_encoded = pd.get_dummies(df, columns=existing_nominal_cols, drop_first=False)
        print(f"[5] One-hot encoding sur {len(existing_nominal_cols)} colonnes catégorielles")
        print(f"    Dimensions après encoding : {df_encoded.shape}")

        if 'claim_status' in df_encoded.columns:
            n_claims = df_encoded['claim_status'].sum()
            n_total = len(df_encoded)
            ratio = (n_claims / n_total * 100) if n_total else 0
            print(f"\n[6] Déséquilibre des classes :")
            print(f"    Sinistres (1)     : {int(n_claims):>6} ({ratio:.1f}%)")
            print(f"    Non-sinistres (0) : {n_total - int(n_claims):>6} ({100-ratio:.1f}%)")
            print(f"    Ratio d'imbalance : 1 pour {int((n_total - n_claims) / n_claims)}" if n_claims else "    Ratio d'imbalance : indéterminé")

        X = df_encoded.drop(columns=['claim_status']) if 'claim_status' in df_encoded.columns else df_encoded.copy()
        y = df_encoded['claim_status'] if 'claim_status' in df_encoded.columns else None

        numeric_cols = [
            'subscription_length', 'vehicle_age', 'customer_age', 'region_density',
            'displacement', 'cylinder', 'turning_radius', 'length', 'width',
            'gross_weight', 'torque_nm', 'torque_rpm', 'power_bhp', 'power_rpm', 'airbags'
        ]
        existing_numeric_cols = [col for col in numeric_cols if col in X.columns]
        X_scaled = X.copy()
        if existing_numeric_cols:
            X_scaled[existing_numeric_cols] = self.scaler.fit_transform(X[existing_numeric_cols])
        print(f"\n[7] Normalisation (StandardScaler) sur {len(existing_numeric_cols)} colonnes numériques")

        return df_encoded, X_scaled, y

    def save_preprocessors(self, output_dir='models'):
        """Save scaler and label encoders for later use."""
        os.makedirs(output_dir, exist_ok=True)
        
        with open(os.path.join(output_dir, 'scaler.pkl'), 'wb') as f:
            pickle.dump(self.scaler, f)
        
        with open(os.path.join(output_dir, 'label_encoders.pkl'), 'wb') as f:
            pickle.dump(self.label_encoders, f)
        
        print(f"\nPréprocesseurs sauvegardés dans '{output_dir}'")
    
    def load_preprocessors(self, output_dir='models'):
        """Load scaler and label encoders from disk."""
        with open(os.path.join(output_dir, 'scaler.pkl'), 'rb') as f:
            self.scaler = pickle.load(f)
        
        with open(os.path.join(output_dir, 'label_encoders.pkl'), 'rb') as f:
            self.label_encoders = pickle.load(f)
        
        print(f"Préprocesseurs chargés depuis '{output_dir}'")


def process_pipeline(input_file, output_dir='data'):
    """Complete processing pipeline: load -> clean -> feature-engineer -> transform -> save."""
    processor = DataProcessor()
    
    print("1. Chargement des données...")
    df = processor.load_data(input_file)
    
    print("\n2. Analyse de la qualité...")
    processor.analyze_quality(df)
    
    print("\n3. Nettoyage...")
    df_clean = processor.clean_data(df)
    
    print("\n4. Détection des outliers...")
    processor.detect_outliers(df_clean)
    
    print("\n5. Prétraitement enrichi...")
    df_encoded, X_scaled, y = processor.preprocess_claims(df_clean)
    
    # Sauvegarder les preprocesseurs
    processor.save_preprocessors()
    
    # Sauvegarder les données transformées
    os.makedirs(output_dir, exist_ok=True)
    X_scaled = X_scaled.copy()
    if y is not None:
        X_scaled['claim_status'] = y.values
    X_scaled.to_csv(os.path.join(output_dir, 'data_processed.csv'), index=False)
    df_encoded.to_csv(os.path.join(output_dir, 'data_encoded.csv'), index=False)
    print(f"\nDonnées transformées sauvegardées dans '{output_dir}'")
    
    return processor, X_scaled
