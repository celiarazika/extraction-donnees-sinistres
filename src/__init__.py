"""
Insurance claims data extraction package.
"""

__version__ = "0.1.0"
__author__ = "Data Science Team"

from .data_processor import DataProcessor, process_pipeline
from .model import create_model, train_model, evaluate_model, predict

__all__ = [
    'DataProcessor',
    'process_pipeline',
    'create_model',
    'train_model',
    'evaluate_model',
    'predict'
]
