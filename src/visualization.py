"""
Module for visualizing similarity results between Greek manuscripts.
"""

import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.manifold import MDS, TSNE

class SimilarityVisualizer:
    """Visualize similarity results between Greek manuscripts."""
    
    def __init__(self, output_dir: str = 'visualizations'):
        """
        Initialize the similarity visualizer.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def create_similarity_heatmap(self, similarity_scores: Dict[str, float], 
                                  title: str = 'Similarity Metrics', 
                                  filename: str = 'similarity_heatmap.png') -> Figure:
        """
        Create a heatmap of similarity scores.
        
        Args:
            similarity_scores: Dictionary of similarity scores
            title: Plot title
            filename: Output filename
            
        Returns:
            Matplotlib figure object
        """
        # Filter out the overall similarity which will be displayed separately
        filtered_scores = {k: v for k, v in similarity_scores.items() if k != 'overall_similarity'}
        
        # Sort metrics by category
        categories = {
            'Vocabulary': ['vocabulary_jaccard', 'vocabulary_cosine'],
            'N-grams': ['bigram_jaccard', 'bigram_cosine', 'trigram_jaccard', 'trigram_cosine'],
            'Sentence': ['sentence_length_mean_sim', 'sentence_length_median_sim', 'sentence_length_std_sim'],
            'Style': ['unique_ratio_sim', 'hapax_ratio_sim', 'yule_k_sim', 'word_position_correlation']
        }
        
        # Prepare data for visualization
        metrics = []
        scores = []
        metric_categories = []
        
        for category, category_metrics in categories.items():
            for metric in category_metrics:
                if metric in filtered_scores:
                    metrics.append(metric)
                    scores.append(filtered_scores[metric])
                    metric_categories.append(category)
        
        # Set up colors for different categories
        category_colors = {
            'Vocabulary': '#1f77b4',  # blue
            'N-grams': '#ff7f0e',     # orange
            'Sentence': '#2ca02c',    # green
            'Style': '#d62728'        # red
        }
        
        # Create color list based on categories
        colors = [category_colors[category] for category in metric_categories]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot horizontal bar chart
        y_pos = np.arange(len(metrics))
        ax.barh(y_pos, scores, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([metric.replace('_', ' ').title() for metric in metrics])
        ax.set_xlim(0, 1)
        ax.set_xlabel('Similarity Score (0-1)')
        ax.set_title(title)
        
        # Add overall similarity as text
        overall_sim = similarity_scores.get('overall_similarity', 0.0)
        ax.text(0.5, 1.05, f'Overall Similarity: {overall_sim:.4f}', 
                horizontalalignment='center', 
                verticalalignment='center', 
                transform=ax.transAxes,
                fontsize=14, 
                fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.8))
        
        # Add legend for categories
        import matplotlib.patches as mpatches
        patches = [mpatches.Patch(color=color, label=category) 
                   for category, color in category_colors.items()]
        ax.legend(handles=patches, loc='lower right')
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename))
        
        return fig
    
    def create_radar_chart(self, similarity_scores: Dict[str, float], 
                           title: str = 'Similarity Radar', 
                           filename: str = 'similarity_radar.png') -> Figure:
        """
        Create a radar chart of similarity scores.
        
        Args:
            similarity_scores: Dictionary of similarity scores
            title: Plot title
            filename: Output filename
            
        Returns:
            Matplotlib figure object
        """
        # Filter out the overall similarity which will be displayed separately
        filtered_scores = {k: v for k, v in similarity_scores.items() if k != 'overall_similarity'}
        
        # Group metrics by category for better visualization
        categories = {
            'Vocabulary': ['vocabulary_jaccard', 'vocabulary_cosine'],
            'Bigrams': ['bigram_jaccard', 'bigram_cosine'],
            'Trigrams': ['trigram_jaccard', 'trigram_cosine'],
            'Sentence Length': ['sentence_length_mean_sim', 'sentence_length_median_sim', 'sentence_length_std_sim'],
            'Vocabulary Richness': ['unique_ratio_sim', 'hapax_ratio_sim', 'yule_k_sim'],
            'Word Positioning': ['word_position_correlation']
        }
        
        # Calculate average score for each category
        category_scores = {}
        for category, metrics in categories.items():
            valid_metrics = [m for m in metrics if m in filtered_scores]
            if valid_metrics:
                category_scores[category] = sum(filtered_scores[m] for m in valid_metrics) / len(valid_metrics)
        
        # Set data
        categories = list(category_scores.keys())
        values = list(category_scores.values())
        
        # Number of variables
        N = len(categories)
        
        # Create angle for each category
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Add values to close the loop
        values += values[:1]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        # Plot data
        ax.plot(angles, values, linewidth=2, linestyle='solid', label="Similarity")
        ax.fill(angles, values, alpha=0.25)
        
        # Add category labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        
        # Set y-axis limits
        ax.set_ylim(0, 1)
        
        # Add overall similarity as text
        overall_sim = similarity_scores.get('overall_similarity', 0.0)
        plt.figtext(0.5, 0.1, f'Overall Similarity: {overall_sim:.4f}', 
                    horizontalalignment='center', 
                    fontsize=14, 
                    fontweight='bold')
        
        # Add title
        plt.title(title)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename))
        
        return fig
    
    def create_similarity_report(self, similarity_scores: Dict[str, float], 
                               manuscript1_name: str, 
                               manuscript2_name: str,
                               output_file: str = 'similarity_report.txt') -> None:
        """
        Create a text report of similarity scores.
        
        Args:
            similarity_scores: Dictionary of similarity scores
            manuscript1_name: Name of first manuscript
            manuscript2_name: Name of second manuscript
            output_file: Output filename
        """
        # Get overall similarity
        overall = similarity_scores.get('overall_similarity', 0.0)
        
        # Group metrics by category
        categories = {
            'Vocabulary Similarity': ['vocabulary_jaccard', 'vocabulary_cosine'],
            'N-gram Similarity': ['bigram_jaccard', 'bigram_cosine', 'trigram_jaccard', 'trigram_cosine'],
            'Sentence Structure': ['sentence_length_mean_sim', 'sentence_length_median_sim', 'sentence_length_std_sim'],
            'Stylistic Features': ['unique_ratio_sim', 'hapax_ratio_sim', 'yule_k_sim', 'word_position_correlation']
        }
        
        # Generate report
        with open(os.path.join(self.output_dir, output_file), 'w') as f:
            f.write(f'Similarity Report: {manuscript1_name} vs {manuscript2_name}\n')
            f.write('=' * 60 + '\n\n')
            
            f.write(f'Overall Similarity Score: {overall:.4f}\n')
            f.write('-' * 60 + '\n\n')
            
            # Interpret overall similarity
            if overall >= 0.8:
                interpretation = "The manuscripts are extremely similar, suggesting very close relationships."
            elif overall >= 0.6:
                interpretation = "The manuscripts show high similarity, indicating strong influences."
            elif overall >= 0.4:
                interpretation = "The manuscripts have moderate similarity, with some shared characteristics."
            elif overall >= 0.2:
                interpretation = "The manuscripts show low similarity, with limited shared characteristics."
            else:
                interpretation = "The manuscripts are very different, suggesting distinct origins."
                
            f.write(f'Interpretation: {interpretation}\n\n')
            
            # Write detailed scores by category
            f.write('Detailed Similarity Scores\n')
            f.write('-' * 60 + '\n\n')
            
            for category, metrics in categories.items():
                f.write(f'{category}:\n')
                
                for metric in metrics:
                    if metric in similarity_scores:
                        pretty_metric = metric.replace('_', ' ').title()
                        score = similarity_scores[metric]
                        f.write(f'  - {pretty_metric}: {score:.4f}\n')
                        
                f.write('\n')
                
            # Add interpretation guidelines
            f.write('Interpretation Guidelines\n')
            f.write('-' * 60 + '\n\n')
            f.write('0.0-0.2: Very Low Similarity\n')
            f.write('0.2-0.4: Low Similarity\n')
            f.write('0.4-0.6: Moderate Similarity\n')
            f.write('0.6-0.8: High Similarity\n')
            f.write('0.8-1.0: Very High Similarity\n')
    
    def visualize_all(self, similarity_scores: Dict[str, float], 
                      manuscript1_name: str, 
                      manuscript2_name: str) -> None:
        """
        Create all visualizations and reports.
        
        Args:
            similarity_scores: Dictionary of similarity scores
            manuscript1_name: Name of first manuscript
            manuscript2_name: Name of second manuscript
        """
        # Create title
        title = f'Similarity: {manuscript1_name} vs {manuscript2_name}'
        
        # Create heatmap
        self.create_similarity_heatmap(
            similarity_scores, 
            title=f'Similarity Metrics: {manuscript1_name} vs {manuscript2_name}',
            filename=f'similarity_heatmap_{manuscript1_name}_vs_{manuscript2_name}.png'
        )
        
        # Create radar chart
        self.create_radar_chart(
            similarity_scores, 
            title=f'Similarity Radar: {manuscript1_name} vs {manuscript2_name}',
            filename=f'similarity_radar_{manuscript1_name}_vs_{manuscript2_name}.png'
        )
        
        # Create text report
        self.create_similarity_report(
            similarity_scores, 
            manuscript1_name, 
            manuscript2_name,
            output_file=f'similarity_report_{manuscript1_name}_vs_{manuscript2_name}.txt'
        ) 