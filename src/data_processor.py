"""
Data processing module for insurance claims extraction.
Handles loading, cleaning, encoding, and normalization of data.
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder


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
        print("="*60)
        print("ANALYSE DE LA QUALITÉ DES DONNÉES")
        print("="*60)
        
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
        print("\n" + "="*60)
        print("DÉTECTION DES VALEURS ABERRANTES")
        print("="*60)
        
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
    
    def transform_data(self, df):
        """Encode categorical variables and normalize numeric variables."""
        print("\n" + "="*60)
        print("TRANSFORMATION ET NORMALISATION")
        print("="*60)
        
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
    """Complete processing pipeline: load -> clean -> transform -> save."""
    processor = DataProcessor()
    
    print("1. Chargement des données...")
    df = processor.load_data(input_file)
    
    print("\n2. Analyse de la qualité...")
    processor.analyze_quality(df)
    
    print("\n3. Nettoyage...")
    df_clean = processor.clean_data(df)
    
    print("\n4. Détection des outliers...")
    processor.detect_outliers(df_clean)
    
    print("\n5. Transformation...")
    df_transformed = processor.transform_data(df_clean)
    
    # Sauvegarder les preprocesseurs
    processor.save_preprocessors()
    
    # Sauvegarder les données transformées
    os.makedirs(output_dir, exist_ok=True)
    df_transformed.to_csv(os.path.join(output_dir, 'data_processed.csv'), index=False)
    print(f"\nDonnées transformées sauvegardées dans '{output_dir}'")
    
    return processor, df_transformed
