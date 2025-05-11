"""
Greek manuscript comparison package.
"""

from .preprocessing import GreekTextPreprocessor
from .features import FeatureExtractor
from .similarity import SimilarityCalculator
from .multi_comparison import MultipleManuscriptComparison
from .advanced_nlp import AdvancedGreekProcessor

__all__ = [
    'GreekTextPreprocessor',
    'FeatureExtractor',
    'SimilarityCalculator',
    'MultipleManuscriptComparison',
    'AdvancedGreekProcessor'
]

__version__ = '0.1.0' 