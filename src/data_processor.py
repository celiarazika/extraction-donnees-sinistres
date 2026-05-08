"""
Data processor for insurance claims - LLM-optimized.
Transforms raw data into semantic, expert-friendly descriptions.
"""

import re
import numpy as np
import pandas as pd


class DataProcessor:
    """Transforms raw claims data into semantic descriptions for LLM injection."""
    
    def __init__(self):
        """Initialize with metadata mappings."""
        # Segment mapping
        self.segment_map = {
            "A": "Citadine",
            "B": "Berline compacte", 
            "B1": "Berline compacte",
            "B2": "Berline compacte",
            "C": "Berline familiale",
            "C1": "Berline familiale",
            "C2": "Berline familiale / SUV",
            "D": "Berline grande classe",
            "Utility": "Utilitaire"
        }
        
        # Fuel type mapping
        self.fuel_map = {
            "Petrol": "Essence",
            "Diesel": "Diesel",
            "CNG": "Gaz naturel (CNG)",
            "Hybrid": "Hybride"
        }
        
        # Transmission mapping
        self.transmission_map = {
            "Manual": "Manuelle",
            "Automatic": "Automatique"
        }
        
        # Brake type mapping
        self.brake_map = {
            "Disc": "Disques",
            "Drum": "Tambours"
        }
        
        # Steering mapping
        self.steering_map = {
            "Manual": "Manuelle",
            "Power": "Assistée",
            "Electric": "Électrique"
        }
        
        # Region density categories
        self.density_categories = {
            "urban_dense": (20000, float('inf'), "Zone urbaine très dense"),
            "urban": (5000, 20000, "Zone urbaine"),
            "suburban": (1000, 5000, "Zone semi-urbaine"),
            "rural": (0, 1000, "Zone rurale")
        }
    
    def load_data(self, filepath):
        """Load CSV data."""
        return pd.read_csv(filepath)
  
    def clean_data(self, df):
        """Clean data: remove duplicates and handle missing values."""
        df_clean = df.copy()
        
        # Remove duplicates
        df_clean = df_clean.drop_duplicates()
        
        # Fill missing values
        text_cols = df_clean.select_dtypes(include=['object']).columns
        df_clean[text_cols] = df_clean[text_cols].fillna("Inconnu").apply(
            lambda x: x.str.strip() if x.dtype == 'object' else x
        )
        
        num_cols = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[num_cols] = df_clean[num_cols].fillna(0)
        
        return df_clean
    
    def preprocess_claims(self, df):
        """Orchestrate preprocessing. Returns (df, encoders, scaler) tuple."""
        df_final = df.copy()
        
        # Add semantic column for LLM
        df_final['llm_input'] = df_final.apply(self._build_semantic_report, axis=1)
        
        return df_final, {}, None
    
    def _categorize_density(self, density):
        """Convert numeric density to qualitative category."""
        try:
            d = float(density)
            for key, (min_d, max_d, label) in self.density_categories.items():
                if min_d <= d < max_d:
                    return label
            return "Zone inconnue"
        except:
            return "Zone inconnue"
    
    def _extract_power(self, max_power_str):
        """Extract power in bhp from string like '113.45bhp@4000rpm'."""
        try:
            match = re.search(r'([\d.]+)\s*bhp', str(max_power_str))
            return float(match.group(1)) if match else None
        except:
            return None
    
    def _extract_torque(self, max_torque_str):
        """Extract torque in Nm from string like '250Nm@2750rpm'."""
        try:
            match = re.search(r'([\d.]+)\s*Nm', str(max_torque_str))
            return float(match.group(1)) if match else None
        except:
            return None
    
    def _build_safety_profile(self, row):
        """Build semantic safety profile from equipment flags."""
        safety_items = []
        
        # Airbags
        airbags = row.get('airbags', 0)
        if airbags:
            safety_items.append(f"{int(airbags)} airbags")
        
        # NCAP Rating
        ncap = row.get('ncap_rating', 0)
        ncap_label = {0: "Aucune notation", 1: "1⭐", 2: "2⭐", 3: "3⭐", 4: "4⭐", 5: "5⭐"}
        ncap_text = ncap_label.get(int(ncap), f"{int(ncap)} étoiles")
        safety_items.append(f"NCAP: {ncap_text}")
        
        # Key safety features
        safety_features = [
            ('is_esc', "Correcteur de trajectoire (ESC)"),
            ('is_brake_assist', "Assistance au freinage"),
            ('is_speed_alert', "Alerte de vitesse"),
        ]
        
        active_features = []
        for col, label in safety_features:
            if row.get(col) in [1, True, 'Yes']:
                active_features.append(label)
        
        if active_features:
            safety_items.append("Aides: " + ", ".join(active_features))
        else:
            safety_items.append("Aides à la conduite: Aucune")
        
        return " | ".join(safety_items)
    
    def _build_vehicle_profile(self, row):
        """Build semantic vehicle profile."""
        profile_items = []
        
        # Segment
        segment = row.get('segment', 'Inconnu')
        segment_label = self.segment_map.get(segment, segment)
        profile_items.append(f"Segment: {segment_label}")
        
        # Fuel type
        fuel = row.get('fuel_type', 'Inconnu')
        fuel_label = self.fuel_map.get(fuel, fuel)
        profile_items.append(f"Carburant: {fuel_label}")
        
        # Transmission
        trans = row.get('transmission_type', 'Inconnu')
        trans_label = self.transmission_map.get(trans, trans)
        profile_items.append(f"Transmission: {trans_label}")
        
        # Power
        power = self._extract_power(row.get('max_power', ''))
        if power:
            profile_items.append(f"Puissance: {power:.0f} bhp")
        
        # Rear brakes
        brakes = row.get('rear_brakes_type', 'Inconnu')
        brake_label = self.brake_map.get(brakes, brakes)
        profile_items.append(f"Freins arrière: {brake_label}")
        
        return " | ".join(profile_items)
    
    def _build_driver_profile(self, row):
        """Build semantic driver/holder profile."""
        profile_items = []
        
        # Age
        age = row.get('customer_age', 'Inconnu')
        profile_items.append(f"Âge: {int(age) if age != 'Inconnu' else 'Inconnu'} ans")
        
        # Subscription length
        sub_length = row.get('subscription_length', 0)
        profile_items.append(f"Ancienneté abonnement: {sub_length:.1f} mois")
        
        # Region density
        density = row.get('region_density', 0)
        density_cat = self._categorize_density(density)
        profile_items.append(f"Zone: {density_cat}")
        
        return " | ".join(profile_items)
    
    def _build_semantic_report(self, row):
        """
        Build complete semantic report for a single claim.
        Outputs a structured bullet-point report, ready for LLM injection.
        """
        vehicle_age = row.get('vehicle_age', 0)
        
        report = []
        report.append("=== PROFIL ASSURÉ ===")
        report.append(self._build_driver_profile(row))
        
        report.append("\n=== VÉHICULE ===")
        report.append(f"Âge du véhicule: {vehicle_age:.1f} ans")
        report.append(self._build_vehicle_profile(row))
        
        report.append("\n=== SÉCURITÉ ET ÉQUIPEMENT ===")
        report.append(self._build_safety_profile(row))
        
        # Risk assessment
        report.append("\n=== ANALYSE DE RISQUE ===")
        risk_factors = []
        
        # Vehicle age risk
        if vehicle_age > 10:
            risk_factors.append("Véhicule ancien (>10 ans): usure mécanique possible")
        elif vehicle_age < 1:
            risk_factors.append("Véhicule très récent: données de conduite limitées")
        
        # Safety rating risk
        ncap = int(row.get('ncap_rating', 0))
        if ncap <= 2:
            risk_factors.append("Note NCAP faible: protection passive limitée")
        
        # Safety equipment risk
        esc = row.get('is_esc', 0)
        brake_assist = row.get('is_brake_assist', 0)
        if esc not in [1, True, 'Yes'] and brake_assist not in [1, True, 'Yes']:
            risk_factors.append("Absence d'aides modernes à la conduite")
        
        # Brake type risk
        rear_brakes = row.get('rear_brakes_type', '')
        if rear_brakes == 'Drum':
            risk_factors.append("Freins arrière à tambour: efficacité réduite")
        
        # Driver age risk
        driver_age = int(row.get('customer_age', 0))
        if driver_age < 25:
            risk_factors.append("Conducteur jeune: expérience de conduite limitée")
        elif driver_age > 75:
            risk_factors.append("Conducteur âgé: capacités réactionnelles réduites")
        
        if risk_factors:
            for factor in risk_factors:
                report.append(f"⚠️ {factor}")
        else:
            report.append("✓ Profil de risque standard")
        
        # Claim status
        report.append("\n=== STATUT ===")
        status = row.get('claim_status', 'Inconnu')
        status_label = "Validé" if status in [1, True, 'Yes'] else "Refusé" if status in [0, False, 'No'] else "Inconnu"
        report.append(f"Statut du sinistre: {status_label}")
        
        return "\n".join(report)
    
    def format_for_llm(self, row_dict):
        """
        Format data for LLM injection.
        Accepts dict or Series, returns semantic report.
        """
        # Convert Series to dict if needed
        if isinstance(row_dict, pd.Series):
            row_dict = row_dict.to_dict()
        
        # If already has semantic report, use it
        if 'llm_input' in row_dict and row_dict['llm_input']:
            return row_dict['llm_input']
        
        # Generate semantic report
        return self._build_semantic_report(row_dict)