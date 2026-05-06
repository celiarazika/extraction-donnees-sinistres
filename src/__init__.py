"""
Insurance claims data extraction and LLM-based description generation package.
"""

__version__ = "0.2.0"
__author__ = "Celia IMAKHLOUFEN"

from .data_processor import DataProcessor, process_pipeline
from .model import ClaimsLLMGenerator, create_generator, generate_claim_description

__all__ = [
    'DataProcessor',
    'process_pipeline',
    'ClaimsLLMGenerator',
    'create_generator',
    'generate_claim_description'
]
