import torch
import math
import copy
import numpy as np

def move_to_index(move_str):
    """
    Fast conversion of UCI move string to policy index (0-8191).
    """
    # 97 is ord('a'), 49 is ord('1')
    # src_col = ord(move_str[0]) - 97
    # src_row = ord(move_str[1]) - 49
    src = (ord(move_str[0]) - 97) + (ord(move_str[1]) - 49) * 8
    dst = (ord(move_str[2]) - 97) + (ord(move_str[3]) - 49) * 8
    idx = src * 64 + dst
    
    # Promotion Logic
    if len(move_str) == 5:
        promotion = move_str[4]
        if promotion == 'n': idx += 4096
        elif promotion == 'r': idx += 4096 * 2 # Just an example offset strategy, keeping your logic
        elif promotion == 'b': idx += 4096 * 3 
        # Note: Your original code used `idx += 4096` for all promotions. 
        # I kept your original logic below to ensure compatibility with weights.
        if promotion in ['n', 'r', 'b']:
            return src * 64 + dst + 4096
            
    return idx

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
        
        # Optimization: Pre-calculate constant part
        sqrt_total_visits = math.sqrt(self.visit_count)
        
        for action, child in self.children.items():
            q_value = child.value_sum / child.visit_count if child.visit_count > 0 else 0
            u_value = cpuct * child.prior * sqrt_total_visits / (1 + child.visit_count)
            score = q_value + u_value
            
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
                
        return best_action, best_child

    def expand(self, valid_moves, policy_logits):
        move_probs = {}
        policy_sum = 0
        
        for move_str in valid_moves:
            # Use the optimized index calculation logic inline or via helper
            # Inline is faster for Python loops
            src = (ord(move_str[0]) - 97) + (int(move_str[1]) - 1) * 8
            dst = (ord(move_str[2]) - 97) + (int(move_str[3]) - 1) * 8
            idx = src * 64 + dst
            
            if len(move_str) == 5 and move_str[4] in ['n', 'r', 'b']:
                idx += 4096
            
            if idx < len(policy_logits):
                logit = policy_logits[idx]
            else:
                logit = -10.0 

            prob = math.exp(logit) 
            move_probs[move_str] = prob
            policy_sum += prob
            
        for move in valid_moves:
            if policy_sum > 0:
                normalized_prior = move_probs[move] / policy_sum
            else:
                normalized_prior = 1.0 / len(valid_moves)
            
            next_state = self.state.copy() # Use .copy() instead of deepcopy for speed
            next_state.push(move)
            
            self.children[move] = Node(next_state, parent=self, action_taken=move, prior=normalized_prior)

    def best_action(self):
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
        actions = list(node.children.keys())
        if not actions: return
        noise = np.random.dirichlet([0.3] * len(actions))
        frac = 0.25 
        
        for i, action in enumerate(actions):
            node.children[action].prior = node.children[action].prior * (1 - frac) + noise[i] * frac

    def get_policy_vector(self, root):
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
        root = Node(root_state)
        
        # 1. Evaluate Root
        tensor = torch.from_numpy(root.state.to_tensor()) # Faster than torch.tensor(..., dtype)
        self.input_queue.put((self.worker_id, tensor))
        policy, value = self.output_queue.get()
        valid_moves = root.state.legal_moves()
        root.expand(valid_moves, policy)
        
        self.add_exploration_noise(root)
        
        # 2. Simulations
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
                
            # Efficient tensor creation
            tensor = torch.from_numpy(node.state.to_tensor())
            self.input_queue.put((self.worker_id, tensor))
            policy, value = self.output_queue.get()
            
            valid_moves = node.state.legal_moves()
            node.expand(valid_moves, policy)
            self.backpropagate(search_path, value, node.state.turn_player)
            
        policy_vector = self.get_policy_vector(root)
        
        # 3. Select Action
        if temperature == 0:
            return root.best_action(), policy_vector
        else:
            actions = list(root.children.keys())
            if not actions: return None, policy_vector

            visits = np.array([root.children[a].visit_count for a in actions], dtype=np.float32)
            
            if temperature != 1.0:
                visits = np.power(visits, 1.0 / temperature)
            
            visit_sum = np.sum(visits)
            if visit_sum == 0:
                probs = np.ones(len(visits)) / len(visits)
            else:
                probs = visits / visit_sum
            
            chosen_action = np.random.choice(actions, p=probs)
            return chosen_action, policy_vector

    def backpropagate(self, path, value, turn_perspective):
        for node in reversed(path):
            node.visit_count += 1
            if node.state.turn_player == turn_perspective:
                node.value_sum += value
            else:
                node.value_sum -= value