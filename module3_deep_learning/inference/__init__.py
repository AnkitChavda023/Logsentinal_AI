"""
Inference Module for LogSentinel

Contains the prediction service for real-time inference.
"""

from .predictor_service import PredictorService, PredictionRequest, PredictionResponse

__all__ = [
    "PredictorService",
    "PredictionRequest",
    "PredictionResponse"
]
