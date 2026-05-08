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
