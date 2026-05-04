"""
LLM-based text generation for insurance claims.
Generates descriptions of insurance claims from structured data.
"""

import os
from typing import Dict, List
import numpy as np


class ClaimsLLMGenerator:
    """Generates claim descriptions using a Language Model."""
    
    def __init__(self, model_name: str = "gpt2"):
        """
        Initialize the LLM generator.
        
        Args:
            model_name: Name of the model to use
                - "gpt2": Free, local model (default)
                - "openai": Requires OPENAI_API_KEY environment variable
                - "hf-model": HuggingFace model
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load the LLM model."""
        print(f"Loading {self.model_name}...")
        
        if self.model_name == "gpt2":
            from transformers import GPT2Tokenizer, GPT2LMHeadModel
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            self.model = GPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=self.tokenizer.eos_token_id)
            print("✅ GPT-2 loaded successfully")
        
        elif self.model_name == "openai":
            import openai
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables")
            openai.api_key = api_key
            print("✅ OpenAI API configured")
        
        else:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            print(f"✅ {self.model_name} loaded successfully")
    
    def create_prompt(self, claim_data: Dict) -> str:
        """
        Create a prompt from structured claim data.
        
        Args:
            claim_data: Dictionary with claim information
        
        Returns:
            Formatted prompt for the LLM
        """
        prompt = "Écris une brève description d'un sinistre d'assurance basée sur ces données:\n\n"
        
        for key, value in claim_data.items():
            prompt += f"- {key}: {value}\n"
        
        prompt += "\nDescription du sinistre:"
        return prompt
    
    def generate(self, claim_data: Dict, max_length: int = 100, num_beams: int = 4) -> str:
        """
        Generate a description for a claim.
        
        Args:
            claim_data: Dictionary with claim information
            max_length: Maximum length of generated text
            num_beams: Number of beams for beam search
        
        Returns:
            Generated claim description
        """
        if self.model_name == "openai":
            return self._generate_openai(claim_data)
        else:
            return self._generate_local(claim_data, max_length, num_beams)
    
    def _generate_local(self, claim_data: Dict, max_length: int, num_beams: int) -> str:
        """Generate using local model (GPT-2 or HuggingFace)."""
        prompt = self.create_prompt(claim_data)
        
        # Tokenize
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        
        # Generate
        output = self.model.generate(
            input_ids,
            max_length=max_length + len(input_ids[0]),
            num_beams=num_beams,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            no_repeat_ngram_size=2
        )
        
        # Decode
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Extract only the generated part (remove prompt)
        return generated_text.replace(prompt, "").strip()
    
    def _generate_openai(self, claim_data: Dict) -> str:
        """Generate using OpenAI API."""
        import openai
        
        prompt = self.create_prompt(claim_data)
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Tu es un expert en assurance. Fournis des descriptions courtes et claires de sinistres."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
            temperature=0.7
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


def create_generator(model_name: str = "gpt2") -> ClaimsLLMGenerator:
    """Factory function to create an LLM generator."""
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
