"""
Region-Aware Fraud Detection Package

Multi-step fraud detection that segments documents first,
then applies analyzers only to comparable regions.
"""

from .region_analyzer import analyze_document

__all__ = ['analyze_document']
