"""
LLM-based text generation for insurance claims.
Generates descriptions of insurance claims from structured data.
"""

import os
from typing import Dict, List
import numpy as np


class ClaimsLLMGenerator:
    """Generates claim descriptions using a Language Model."""
    
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
            model="phi3.5",  # Changed from mistral for speed (2-3x faster)
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
    
    def generate_synthetic_data(self, schema_context: str, num_rows: int) -> str:
            """
            Demande au LLM de générer N lignes de CSV synthétique en utilisant
            une structure stricte par l'exemple (Few-Shot Prompting).
            """
            
            # L'en-tête exact basé sur votre base de données
            columns = "policy_id,subscription_length,vehicle_age,customer_age,region_code,region_density,segment,model,fuel_type,max_torque,max_power,engine_type,airbags,is_esc,is_adjustable_steering,is_tpms,is_parking_sensors,is_parking_camera,rear_brakes_type,displacement,cylinder,transmission_type,steering_type,turning_radius,length,width,gross_weight,is_front_fog_lights,is_rear_window_wiper,is_rear_window_washer,is_rear_window_defogger,is_brake_assist,is_power_door_locks,is_central_locking,is_power_steering,is_driver_seat_height_adjustable,is_day_night_rear_view_mirror,is_ecw,is_speed_alert,ncap_rating,claim_status"
            
            # Un exemple parfait pour forcer le modèle à comprendre le format attendu
            example_row="POL045360,9.3,1.2,41,C8,8794,C2,M4,Diesel,250Nm@2750rpm,113.45bhp@4000rpm,1.5 L U2 CRDi,6,Yes,Yes,Yes,Yes,Yes,Disc,1493,4,Automatic,Power,5.2,4300,1790,1720,Yes,Yes,Yes,Yes,Yes,Yes,Yes,Yes,Yes,No,Yes,Yes,3,0"
            prompt = f"""Tu es un script automatisé de génération de données. Ta SEULE fonction est d'écrire du texte au format CSV.
    Tu dois générer EXACTEMENT {num_rows} lignes de données de sinistres automobiles.

    RÈGLES ABSOLUES ET STRICTES :
    1. La toute première ligne de ta réponse DOIT être cet en-tête exact :
    {columns}
    2. Les lignes suivantes doivent ressembler à cet exemple, mais avec des valeurs inventées et réalistes :
    {example_row}
    3. N'écris AUCUN texte avant. N'écris AUCUN texte après.
    4. Ne mets pas de balises comme ```csv ou ```.
    5. Juste les données brutes séparées par des virgules.

    Génère maintenant l'en-tête et les {num_rows} lignes :"""

            # Appel au LLM
            response = self.client.chat.completions.create(
                model="phi3.5", # Assurez-vous d'utiliser le modèle que vous avez pull (ex: llama3.2:1b)
                messages=[
                    {"role": "system", "content": "Tu es un terminal de commande. Tu ne réponds qu'avec du format CSV pur. Aucune phrase, aucune politesse."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3, # Baisé à 0.3 pour éviter qu'il ne devienne "créatif" et n'hallucine des textes
                max_tokens=1500  # Limite pour éviter qu'il ne boucle à l'infini
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
