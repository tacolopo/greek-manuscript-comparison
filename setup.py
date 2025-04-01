#!/usr/bin/env python3
"""
Setup script for Greek Manuscript Comparison Tool.
"""

from setuptools import setup, find_packages

setup(
    name="greek_manuscript_comparison",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.3",
        "scipy>=1.10.1",
        "scikit-learn>=1.2.2",
        "nltk>=3.8.1",
        "gensim>=4.3.1",
        "matplotlib>=3.7.1",
        "pandas>=2.0.1",
        "spacy>=3.5.3",
        "tqdm>=4.65.0",
        "cltk>=1.1.6",
    ],
    entry_points={
        'console_scripts': [
            'compare-manuscripts=src.compare_manuscripts:main',
        ],
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="A tool for analyzing and comparing Greek manuscripts",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
) 