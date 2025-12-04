import torch
import math
import copy
import numpy as np

# Node class MUST be defined before MCTSWorker uses it
class Node:
    def __init__(self, state, parent=None, action_taken=None, prior=0):
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        
        self.children = {}  
        self.visit_count = 0
        self.value_sum = 0
        self.prior = prior  
        
    def is_expanded(self):
        return len(self.children) > 0

    def select_child(self, cpuct):
        best_score = -float('inf')
        best_action = None
        best_child = None
        
        for action, child in self.children.items():
            # Q-value (Exploitation)
            q_value = child.value_sum / child.visit_count if child.visit_count > 0 else 0
            
            # U-value (Exploration)
            # PUCT: Prior * (sqrt(ParentVisits) / (1 + ChildVisits))
            u_value = cpuct * child.prior * math.sqrt(self.visit_count) / (1 + child.visit_count)
            
            score = q_value + u_value
            
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
                
        return best_action, best_child

    def expand(self, valid_moves, policy_logits):
        """
        Maps Neural Network output (8192 logits) to valid moves.
        """
        move_probs = {}
        policy_sum = 0
        
        for move_str in valid_moves:
            # 1. Base Hash: From-Square to To-Square (0 - 4095)
            src = (ord(move_str[0]) - 97) + (int(move_str[1]) - 1) * 8
            dst = (ord(move_str[2]) - 97) + (int(move_str[3]) - 1) * 8
            idx = src * 64 + dst
            
            # 2. Promotion Logic
            if len(move_str) == 5:
                promotion_type = move_str[4]
                if promotion_type in ['n', 'r', 'b']:
                    idx += 4096
            
            # 3. Safety Check
            if idx < len(policy_logits):
                logit = policy_logits[idx]
            else:
                logit = -10.0 

            prob = math.exp(logit) 
            move_probs[move_str] = prob
            policy_sum += prob
            
        # 4. Normalize
        for move in valid_moves:
            if policy_sum > 0:
                normalized_prior = move_probs[move] / policy_sum
            else:
                normalized_prior = 1.0 / len(valid_moves)
            
            next_state = copy.deepcopy(self.state)
            next_state.push(move)
            
            self.children[move] = Node(next_state, parent=self, action_taken=move, prior=normalized_prior)

    def best_action(self):
        """Returns action with most visits (Exploitation)."""
        most_visits = -1
        best_action = None
        for action, child in self.children.items():
            if child.visit_count > most_visits:
                most_visits = child.visit_count
                best_action = action
        return best_action

class MCTSWorker:
    def __init__(self, worker_id, input_queue, output_queue, simulations=50):
        self.worker_id = worker_id
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.simulations = simulations
        self.cpu = 1.0 
    
    def add_exploration_noise(self, node):
        """
        Adds Dirichlet noise to the prior probabilities of the root node's children.
        This forces the MCTS to explore different moves in self-play.
        """
        actions = list(node.children.keys())
        noise = np.random.dirichlet([0.3] * len(actions))
        frac = 0.25 # 25% noise, 75% original policy
        
        for i, action in enumerate(actions):
            node.children[action].prior = node.children[action].prior * (1 - frac) + noise[i] * frac

    def get_policy_vector(self, root):
        """Converts visit counts to probability vector."""
        policy_vector = np.zeros(8192, dtype=np.float32)
        visit_sum = sum(child.visit_count for child in root.children.values())
        
        if visit_sum == 0: return policy_vector
            
        for action_uci, child in root.children.items():
            src = (ord(action_uci[0]) - 97) + (int(action_uci[1]) - 1) * 8
            dst = (ord(action_uci[2]) - 97) + (int(action_uci[3]) - 1) * 8
            idx = src * 64 + dst
            if len(action_uci) == 5 and action_uci[4] in ['n', 'r', 'b']:
                idx += 4096
            if idx < 8192:
                policy_vector[idx] = child.visit_count / visit_sum
                
        return policy_vector

    def search(self, root_state, temperature=1.0):
        """
        Runs MCTS simulations and selects an action.
        
        Args:
            root_state: The current ChessGame state.
            temperature (float): Controls Exploration vs Exploitation.
                - 1.0: Exploration (Proportional to visit counts)
                - 0.0: Exploitation (Strictly max visits)
                - 0.1 - 0.9: Mixed strategies (Sharpened distribution)
        """
        root = Node(root_state)
        
        # 1. Evaluate Root immediately
        tensor_numpy = root.state.to_tensor()
        tensor = torch.tensor(tensor_numpy, dtype=torch.float32)
        self.input_queue.put((self.worker_id, tensor))
        policy, value = self.output_queue.get()
        valid_moves = root.state.legal_moves()
        root.expand(valid_moves, policy)
        
        # 2. Add Noise (Critical for diversity)
        self.add_exploration_noise(root)
        
        # 3. Run Simulations
        for _ in range(self.simulations):
            node = root
            search_path = [node]
            
            while node.is_expanded():
                action, node = node.select_child(self.cpu)
                search_path.append(node)
                
            if node.state.is_over:
                reward = node.state.get_reward_for_turn(node.state.turn_player)
                self.backpropagate(search_path, reward, node.state.turn_player)
                continue
                
            tensor_numpy = node.state.to_tensor() 
            tensor = torch.tensor(tensor_numpy, dtype=torch.float32)
            self.input_queue.put((self.worker_id, tensor))
            policy, value = self.output_queue.get()
            valid_moves = node.state.legal_moves()
            node.expand(valid_moves, policy)
            self.backpropagate(search_path, value, node.state.turn_player)
            
        policy_vector = self.get_policy_vector(root)
        
        # 4. Select Action based on Temperature
        if temperature == 0:
            # Exploitation: Strictly pick the most visited node
            return root.best_action(), policy_vector
        else:
            # Exploration: Sample from the distribution of visits
            # Probability P(a) ~ (Visits)^(1/Temperature)
            actions = list(root.children.keys())
            visits = np.array([root.children[a].visit_count for a in actions])
            
            if len(actions) == 0:
                return None, policy_vector # Should not happen if game not over

            if temperature != 1.0:
                # Sharpen or flatten the distribution
                visits = visits ** (1.0 / temperature)
            
            # Normalize to probabilities
            if np.sum(visits) == 0:
                probs = np.ones(len(visits)) / len(visits)
            else:
                probs = visits / np.sum(visits)
            
            chosen_action = np.random.choice(actions, p=probs)
            return chosen_action, policy_vector

    def backpropagate(self, path, value, turn_perspective):
        for node in reversed(path):
            node.visit_count += 1
            if node.state.turn_player == turn_perspective:
                node.value_sum += value
            else:
                node.value_sum -= value