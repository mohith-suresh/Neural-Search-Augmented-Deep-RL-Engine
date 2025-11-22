#!/usr/bin/env python3
"""
MCTS Tree Implementation for Chess AI
======================================

Project: EE542 - Deconstructing AlphaZero's Success
Implementation: Monte Carlo Tree Search with UCB exploration

Key Features:
- UCB1 exploration-exploitation balance
- Neural network policy and value guidance
- Efficient tree traversal and backpropagation
- Dirichlet noise for exploration at root
- Virtual loss for parallel MCTS support

Algorithm:
1. Selection: UCB traversal from root to leaf
2. Expansion: Create child nodes for legal moves
3. Evaluation: Neural network forward pass
4. Backpropagation: Update statistics up the tree

References:
- Silver et al. (2017): AlphaZero MCTS with neural guidance
- Browne et al. (2012): Survey of Monte Carlo Tree Search Methods
"""

import math
import numpy as np
import chess
from typing import Dict, List, Optional, Tuple


class MCTSNode:
    """
    Single node in MCTS tree representing a board position.

    Attributes:
        board: chess.Board object for this position
        parent: Parent MCTSNode (None for root)
        move: Move that led to this node from parent
        children: Dict[chess.Move, MCTSNode] of explored children
        visit_count: Number of times this node was visited
        total_value: Cumulative value from backpropagation
        prior_prob: Prior probability from policy network
        is_expanded: Whether children have been created
    """

    def __init__(
        self,
        board: chess.Board,
        parent: Optional['MCTSNode'] = None,
        move: Optional[chess.Move] = None,
        prior: float = 0.0
    ):
        """
        Initialize MCTS node.

        Args:
            board: Chess board position
            parent: Parent node in tree
            move: Move from parent to this node
            prior: Prior probability from policy network
        """
        self.board = board.copy()
        self.parent = parent
        self.move = move
        self.children: Dict[chess.Move, MCTSNode] = {}

        # MCTS statistics
        self.visit_count = 0
        self.total_value = 0.0
        self.prior_prob = prior

        # Expansion state
        self.is_expanded = False

        # Virtual loss for parallel MCTS
        self.virtual_loss = 0

    def is_leaf(self) -> bool:
        """Check if node is a leaf (unexpanded)."""
        return not self.is_expanded

    def is_root(self) -> bool:
        """Check if node is root."""
        return self.parent is None

    def value(self) -> float:
        """
        Get mean value of this node.

        Returns:
            Mean value (wins - losses) / visits
        """
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def ucb_score(self, c_puct: float = 1.0, parent_visits: Optional[int] = None) -> float:
        """
        Calculate Upper Confidence Bound score for node selection.

        UCB = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))

        where:
        - Q(s,a): Mean action value (exploitation)
        - P(s,a): Prior probability from policy network
        - N(s): Parent visit count
        - N(s,a): This node's visit count
        - c_puct: Exploration constant (balance exploration/exploitation)

        Args:
            c_puct: Exploration constant (higher = more exploration)
            parent_visits: Parent node visit count

        Returns:
            UCB score for this node
        """
        if parent_visits is None:
            parent_visits = self.parent.visit_count if self.parent else 1

        # Q-value (exploitation term)
        q_value = self.value()

        # U-value (exploration term)
        # U = c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        u_value = (
            c_puct * self.prior_prob *
            math.sqrt(parent_visits) / (1 + self.visit_count)
        )

        return q_value + u_value

    def select_child(self, c_puct: float = 1.0) -> 'MCTSNode':
        """
        Select child with highest UCB score.

        Args:
            c_puct: Exploration constant

        Returns:
            Child node with highest UCB score
        """
        best_score = -float('inf')
        best_child = None

        for child in self.children.values():
            score = child.ucb_score(c_puct, self.visit_count)
            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def expand(self, policy_probs: Dict[chess.Move, float]) -> None:
        """
        Expand node by creating children for all legal moves.

        Args:
            policy_probs: Dict mapping legal moves to prior probabilities
        """
        if self.is_expanded:
            return

        for move in self.board.legal_moves:
            # Get prior probability (default to small value if not in policy)
            prior = policy_probs.get(move, 1e-8)

            # Create child board
            child_board = self.board.copy()
            child_board.push(move)

            # Create child node
            child_node = MCTSNode(
                board=child_board,
                parent=self,
                move=move,
                prior=prior
            )

            self.children[move] = child_node

        self.is_expanded = True

    def backpropagate(self, value: float) -> None:
        """
        Backpropagate value up the tree.

        Value is negated at each level (zero-sum game).

        Args:
            value: Value to backpropagate (from current player's perspective)
        """
        node = self
        while node is not None:
            node.visit_count += 1
            node.total_value += value

            # Negate value for opponent (zero-sum game)
            value = -value

            node = node.parent

    def add_dirichlet_noise(self, alpha: float = 0.3, epsilon: float = 0.25) -> None:
        """
        Add Dirichlet noise to prior probabilities at root for exploration.

        This encourages exploration at the root node during self-play.
        P(s,a) = (1 - epsilon) * P(s,a) + epsilon * Dirichlet(alpha)

        Args:
            alpha: Dirichlet concentration parameter (0.3 for chess)
            epsilon: Mixing weight for noise (0.25 standard)
        """
        if not self.children:
            return

        # Generate Dirichlet noise
        noise = np.random.dirichlet([alpha] * len(self.children))

        # Apply noise to children
        for child, noise_val in zip(self.children.values(), noise):
            child.prior_prob = (1 - epsilon) * child.prior_prob + epsilon * noise_val

    def get_visit_distribution(self, temperature: float = 1.0) -> Dict[chess.Move, float]:
        """
        Get move probabilities based on visit counts.

        With temperature:
        - T = 0: Deterministic (max visits)
        - T = 1: Proportional to visits
        - T > 1: More uniform

        Args:
            temperature: Temperature for probability distribution

        Returns:
            Dict mapping moves to probabilities
        """
        if not self.children:
            return {}

        moves = []
        visits = []

        for move, child in self.children.items():
            moves.append(move)
            visits.append(child.visit_count)

        visits = np.array(visits, dtype=np.float64)

        if temperature == 0:
            # Deterministic: pick most visited
            probs = np.zeros_like(visits)
            probs[np.argmax(visits)] = 1.0
        else:
            # Apply temperature
            visits = visits ** (1.0 / temperature)
            probs = visits / np.sum(visits)

        return dict(zip(moves, probs))

    def best_move(self) -> Optional[chess.Move]:
        """
        Get move with highest visit count.

        Returns:
            Best move according to MCTS search
        """
        if not self.children:
            return None

        return max(self.children.items(), key=lambda x: x[1].visit_count)[0]

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"MCTSNode(visits={self.visit_count}, "
            f"value={self.value():.3f}, "
            f"prior={self.prior_prob:.3f}, "
            f"children={len(self.children)})"
        )


class MCTS:
    """
    Monte Carlo Tree Search implementation with neural network guidance.

    Implements AlphaZero-style MCTS with:
    - Neural network policy and value evaluation
    - UCB-based tree traversal
    - Dirichlet noise for root exploration
    - Temperature-based move selection
    """

    def __init__(
        self,
        neural_net,
        num_simulations: int = 800,
        c_puct: float = 1.0,
        temperature: float = 1.0,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25
    ):
        """
        Initialize MCTS.

        Args:
            neural_net: Neural network with predict(board) -> (policy, value)
            num_simulations: Number of MCTS simulations per move
            c_puct: Exploration constant for UCB
            temperature: Temperature for move selection
            dirichlet_alpha: Dirichlet noise alpha parameter
            dirichlet_epsilon: Dirichlet noise mixing weight
        """
        self.neural_net = neural_net
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.temperature = temperature
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon

    def search(self, board: chess.Board, add_noise: bool = False) -> Dict[chess.Move, float]:
        """
        Run MCTS search from given position.

        Args:
            board: Starting chess position
            add_noise: Whether to add Dirichlet noise at root (for self-play)

        Returns:
            Dict mapping moves to visit-based probabilities
        """
        # Create root node
        root = MCTSNode(board)

        # Expand root with neural network
        policy_probs, _ = self.neural_net.predict(root.board)
        root.expand(policy_probs)

        # Add exploration noise if requested (for self-play)
        if add_noise:
            root.add_dirichlet_noise(self.dirichlet_alpha, self.dirichlet_epsilon)

        # Run simulations
        for _ in range(self.num_simulations):
            self._simulate(root)

        # Return visit distribution
        return root.get_visit_distribution(self.temperature)

    def _simulate(self, root: MCTSNode) -> None:
        """
        Run single MCTS simulation.

        1. Selection: Traverse tree using UCB until leaf
        2. Expansion: Create children at leaf
        3. Evaluation: Neural network evaluation
        4. Backpropagation: Update statistics

        Args:
            root: Root node to start simulation
        """
        node = root
        search_path = [node]

        # 1. Selection: Traverse to leaf
        while not node.is_leaf() and not node.board.is_game_over():
            node = node.select_child(self.c_puct)
            search_path.append(node)

        # 2. Terminal check
        if node.board.is_game_over():
            # Use actual game result
            result = node.board.result()
            if result == "1-0":
                value = 1.0 if node.board.turn == chess.BLACK else -1.0
            elif result == "0-1":
                value = 1.0 if node.board.turn == chess.WHITE else -1.0
            else:
                value = 0.0  # Draw
        else:
            # 3. Expansion and Evaluation
            policy_probs, value = self.neural_net.predict(node.board)
            node.expand(policy_probs)

        # 4. Backpropagation
        for n in reversed(search_path):
            n.backpropagate(value)
            value = -value  # Negate for opponent

    def get_move_probabilities(
        self,
        board: chess.Board,
        add_noise: bool = False
    ) -> Tuple[Dict[chess.Move, float], chess.Move]:
        """
        Get move probabilities and best move from MCTS search.

        Args:
            board: Chess position to search
            add_noise: Add Dirichlet noise for exploration

        Returns:
            (move_probabilities, best_move)
        """
        move_probs = self.search(board, add_noise=add_noise)

        # Sample move according to probabilities
        if self.temperature == 0:
            # Deterministic
            best_move = max(move_probs.items(), key=lambda x: x[1])[0]
        else:
            # Stochastic
            moves = list(move_probs.keys())
            probs = list(move_probs.values())
            best_move = np.random.choice(moves, p=probs)

        return move_probs, best_move
