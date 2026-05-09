"""
LLM-based text generation for insurance claims.
"""

import os
from typing import Dict, List
import numpy as np


class ClaimsLLMGenerator:
    """Generates claim using a Language Model."""
    
    def __init__(self, model_name: str = "ollama"):
        """
        Initialize the LLM generator.
        
        Args:
            model_name: Name of the model to use (ollama - local API)
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.client = None
        self._load_model()
    
    def _load_model(self):
        """Load Ollama LLM via OpenAI-compatible API."""
        if self.model_name != "ollama":
            raise ValueError(f"Only 'ollama' model is supported, got {self.model_name}")
        
        from openai import OpenAI
        # Ollama exposes an OpenAI-compatible API on localhost:11434
        self.client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama"  # Ollama doesn't require a real API key
        )
        print("Ollama API configured (local - FREE)")
    
    def create_prompt(self, claim_data) -> str:
        """
        Create a prompt from structured claim data.
        Accepts both dict and pre-formatted string input.
        
        Args:
            claim_data: Dictionary with claim information OR pre-formatted string
        
        Returns:
            Formatted prompt for the LLM
        """
        # If already formatted as string, use it directly
        if isinstance(claim_data, str):
            claim_details = claim_data
        else:
            # Format claim data from dict
            claim_details = "\n".join(f"- {key}: {value}" for key, value in claim_data.items())
        
        prompt = f"""Tu es un expert en sinistres d'assurance automobile. Analyse ces données et rédige une description concise et précise du dossier:

DONNÉES DU DOSSIER:
{claim_details}

Rédige une description professionnelle de 50-100 mots basée UNIQUEMENT sur les informations fourni au-dessus. Pas d'inventions, pas d'hypothèses:
"""
        return prompt
    
    def generate(self, claim_data: Dict, max_length: int = 300) -> str:
        """
        Generate a description for a claim using Ollama.
        
        Args:
            claim_data: Dictionary with claim information
            max_length: Maximum length of generated text (ignored, for API compatibility)
        
        Returns:
            Generated claim description
        """
        return self._generate_ollama(claim_data)
    
    
    def _generate_ollama(self, claim_data: Dict) -> str:
        """Generate using Ollama API (local, free). Optimized for detailed output."""
        prompt = self.create_prompt(claim_data)
        
        response = self.client.chat.completions.create(
            model="llama3.2:1b",  # Changed from mistral for speed (2-3x faster)
            messages=[
                {"role": "system", "content": "You are a senior insurance claims expert with 20 years of experience. Provide detailed, accurate, and professional descriptions of insurance claims including all relevant elements for expertise."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,  # Reduced from 500 for faster generation
            temperature=0.2,
            top_p=0.9
        )
        
        return response.choices[0].message.content.strip()
    
    def generate_batch(self, claims_data: List[Dict]) -> List[str]:
        """
        Generate descriptions for multiple claims.
        
        Args:
            claims_data: List of claim dictionaries
        
        Returns:
            List of generated descriptions
        """
        descriptions = []
        for i, claim in enumerate(claims_data):
            print(f"Generating {i+1}/{len(claims_data)}...", end='\r')
            description = self.generate(claim)
            descriptions.append(description)
        print(f"✅ Generated {len(descriptions)} descriptions")
        return descriptions
    
    def generate_synthetic_data(self, schema_context: str, dynamic_examples: str, num_rows: int) -> str:
        """
        Génère du CSV en utilisant du Dynamic Few-Shot Prompting.
        """
        columns = "policy_id,subscription_length,vehicle_age,customer_age,region_code,region_density,segment,model,fuel_type,max_torque,max_power,engine_type,airbags,is_esc,is_adjustable_steering,is_tpms,is_parking_sensors,is_parking_camera,rear_brakes_type,displacement,cylinder,transmission_type,steering_type,turning_radius,length,width,gross_weight,is_front_fog_lights,is_rear_window_wiper,is_rear_window_washer,is_rear_window_defogger,is_brake_assist,is_power_door_locks,is_central_locking,is_power_steering,is_driver_seat_height_adjustable,is_day_night_rear_view_mirror,is_ecw,is_speed_alert,ncap_rating,claim_status"

        prompt = f"""Tu es un générateur de données synthétiques expert. Ta SEULE fonction est d'écrire du texte au format CSV pur.

OBJECTIF : Générer EXACTEMENT {num_rows} lignes de données de sinistres automobiles pour des dossiers AVÉRÉS (claim_status doit TOUJOURS être égal à 1).

CONTEXTE STATISTIQUE (Basé uniquement sur les sinistres réels) :
{schema_context}

RÈGLES DE COHÉRENCE MÉTIER OBLIGATOIRES :
1. 'policy_id' doit commencer par "POL" suivi de 6 chiffres uniques inventés (ex: POL993821).
2. 'claim_status' doit être STRICTEMENT ÉGAL À 1 pour toutes les lignes.
3. Inspire-toi profondément des corrélations visibles dans les exemples fournis ci-dessous pour créer de nouveaux profils réalistes.

FORMAT ATTENDU (Voici des exemples de VRAIS sinistres tirés de la base, utilise-les comme modèles pour la structure et la variance des données) :
{columns}
{dynamic_examples}

INSTRUCTIONS FINALES :
- Renvoie UNIQUEMENT l'en-tête suivi de tes {num_rows} nouvelles lignes générées.
- Ne copie pas exactement les exemples, invente de nouvelles combinaisons logiques.
- Ne mets AUCUNE balise markdown (pas de ```csv).
"""

        response = self.client.chat.completions.create(
            model="mistral", # ou le modèle que vous utilisez
            messages=[
                {"role": "system", "content": "Tu es un terminal qui produit exclusivement du CSV valide. Aucun texte additionnel n'est toléré."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.6, 
            max_tokens=2500,
            top_p=0.9
        )
        
        return response.choices[0].message.content.strip()
    
def create_generator(model_name: str = "ollama") -> ClaimsLLMGenerator:
    """Factory function to create an LLM generator (Ollama only)."""
    return ClaimsLLMGenerator(model_name)


def generate_claim_description(claim_data: Dict, generator: ClaimsLLMGenerator = None) -> str:
    """
    Quick function to generate a single claim description.
    
    Args:
        claim_data: Dictionary with claim information
        generator: Optional pre-loaded generator (creates new one if None)
    
    Returns:
        Generated description
    """
    if generator is None:
        generator = create_generator()
    
    return generator.generate(claim_data)
