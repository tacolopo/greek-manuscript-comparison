"""
Greek Manuscript NLP Analysis Package

Simplified package for natural language processing analysis of ancient Greek texts.
Focus on preprocessing, feature extraction, similarity calculation, and clustering.
"""

from .preprocessing import GreekTextPreprocessor
from .features import FeatureExtractor
from .similarity import SimilarityCalculator
from .multi_comparison import MultipleManuscriptComparison

try:
    from .advanced_nlp import AdvancedGreekProcessor
except ImportError:
    AdvancedGreekProcessor = None

__version__ = "1.0.0"
__author__ = "Greek Manuscript Analysis Team"

__all__ = [
    'GreekTextPreprocessor',
    'FeatureExtractor', 
    'SimilarityCalculator',
    'MultipleManuscriptComparison',
    'AdvancedGreekProcessor'
] 