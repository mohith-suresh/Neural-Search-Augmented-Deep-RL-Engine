"""
MCTS + Self-Play Module for Chess AI

Project: EE542 - Deconstructing AlphaZero's Success

This module provides:
- Monte Carlo Tree Search implementation
- Self-play game generation
- Neural network integration
- Comprehensive evaluation framework
"""

from .mcts_tree import MCTS, MCTSNode
from .neural_mcts_player import NeuralMCTSPlayer, NeuralNetWrapper
from .self_play import SelfPlayWorker, SelfPlayConfig
from .mcts_evaluation import MCTSEvaluator, EvaluationConfig

__all__ = [
    'MCTS',
    'MCTSNode',
    'NeuralMCTSPlayer',
    'NeuralNetWrapper',
    'SelfPlayWorker',
    'SelfPlayConfig',
    'MCTSEvaluator',
    'EvaluationConfig',
]

__version__ = '1.0.0'
__author__ = 'EE542 Team'
