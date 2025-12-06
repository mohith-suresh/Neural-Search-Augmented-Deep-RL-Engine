import torch
import math
import copy
import numpy as np

# --- Constants ---
# Virtual Loss adds "fake" visits to nodes currently being processed by the GPU.
# This discourages other threads in the SAME batch from picking the exact same path.
VIRTUAL_LOSS = 5.0  

def move_to_index(move_str):
    """
    Fast conversion of UCI move string to policy index (0-8191).
    """
    src = (ord(move_str[0]) - 97) + (ord(move_str[1]) - 49) * 8
    dst = (ord(move_str[2]) - 97) + (ord(move_str[3]) - 49) * 8
    idx = src * 64 + dst
    
    # Promotion Logic
    if len(move_str) == 5:
        promotion = move_str[4]
        if promotion == 'q': idx += 4096 
        elif promotion == 'r': idx += 4096 + 64
        elif promotion == 'b': idx += 4096 + 128
        elif promotion == 'n': idx += 4096 + 192
    
    return idx if idx < 8192 else 0

class Node:
    def __init__(self, state, parent=None, prior=0):
        self.state = state
        self.children = {}  
        self.visit_count = 0
        self.value_sum = 0
        self.prior = prior  
        
        # Virtual loss is tracked separately from visit_count
        self.virtual_loss = 0 
        
    def is_expanded(self):
        return len(self.children) > 0

    def value(self):
        # Calculate Q-value including Virtual Loss
        # We add virtual_loss to the denominator to lower the Q-value temporarily
        visits = self.visit_count + self.virtual_loss
        if visits <= 0: return 0
        return self.value_sum / visits

    def select_child(self, cpuct):
        best_score = -float('inf')
        best_action = None
        best_child = None
        
        # Parent visits also include its own virtual loss
        parent_visits = self.visit_count + self.virtual_loss
        sqrt_parent_visits = math.sqrt(max(1, parent_visits))
        
        for action, child in self.children.items():
            # Child visits include its specific virtual loss
            child_visits = child.visit_count + child.virtual_loss
            
            q_value = child.value()
            
            # AlphaZero PUCT Formula
            u_value = cpuct * child.prior * sqrt_parent_visits / (1 + child_visits)
            score = q_value + u_value
            
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
                
        return best_action, best_child

    def expand(self, valid_moves, policy_logits):
        move_probs = {}
        policy_sum = 0
        
        # Softmax over valid moves only
        for move_str in valid_moves:
            idx = move_to_index(move_str)
            logit = policy_logits[idx] if idx < len(policy_logits) else -10.0
            prob = math.exp(logit)
            move_probs[move_str] = prob
            policy_sum += prob
            
        for move in valid_moves:
            if policy_sum > 0:
                normalized_prior = move_probs[move] / policy_sum
            else:
                normalized_prior = 1.0 / len(valid_moves)
                
            next_state = self.state.copy()
            next_state.push(move)
            self.children[move] = Node(next_state, parent=self, prior=normalized_prior)

    def best_action(self):
        # Select purely by real visit count
        most_visits = -1
        best_action = None
        for action, child in self.children.items():
            if child.visit_count > most_visits:
                most_visits = child.visit_count
                best_action = action
        return best_action

class MCTSWorker:
    def __init__(self, worker_id, input_queue, output_queue, simulations=800, batch_size=8):
        self.worker_id = worker_id
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.simulations = simulations
        self.batch_size = batch_size 
        self.cpu = 1.0 

    def get_policy_vector(self, root):
        policy_vector = np.zeros(8192, dtype=np.float32)
        visit_sum = sum(child.visit_count for child in root.children.values())
        if visit_sum == 0: return policy_vector
        for action_uci, child in root.children.items():
            idx = move_to_index(action_uci)
            if idx < 8192: policy_vector[idx] = child.visit_count / visit_sum
        return policy_vector

    def search(self, root_state, temperature=1.0):
        root = Node(root_state)
        
        # 1. Expand Root (Single request to bootstrap)
        tensor = torch.from_numpy(root.state.to_tensor())
        self.input_queue.put((self.worker_id, tensor)) 
        
        # Root request returns single items, not arrays
        policy, value = self.output_queue.get()
        root.expand(root.state.legal_moves(), policy)
        
        self.add_exploration_noise(root)
        
        # 2. Main Loop: Run N simulations in batches
        num_iterations = max(1, self.simulations // self.batch_size)
        
        for _ in range(num_iterations):
            leaves = []
            paths = []
            tensors = []
            
            # --- SELECTION PHASE ---
            for _ in range(self.batch_size):
                node = root
                path = [node]
                
                # Traverse tree
                while node.is_expanded():
                    action, node = node.select_child(self.cpu)
                    path.append(node)
                    
                    # Apply Virtual Loss to this path immediately
                    # This modifies the node state so the next iteration 
                    # in this loop sees these nodes as "visited"
                    node.virtual_loss += VIRTUAL_LOSS
                    node.value_sum -= VIRTUAL_LOSS 
                
                # Check Terminal State
                if node.state.is_over:
                    # Resolve locally, do not send to GPU
                    reward = node.state.get_reward_for_turn(node.state.turn_player)
                    self.backpropagate(path, reward, node.state.turn_player, is_terminal=True)
                else:
                    leaves.append(node)
                    paths.append(path)
                    tensors.append(node.state.to_tensor())

            if not leaves:
                continue

            # --- BATCH INFERENCE PHASE ---
            batch_tensor = torch.from_numpy(np.array(tensors))
            self.input_queue.put((self.worker_id, batch_tensor))
            
            # Wait for response (Arrays of policies and values)
            policies, values = self.output_queue.get()
            
            # --- EXPANSION & BACKPROP PHASE ---
            for i, node in enumerate(leaves):
                path = paths[i]
                policy = policies[i]
                value = values[i]
                
                node.expand(node.state.legal_moves(), policy)
                
                # Backpropagate (and revert Virtual Loss)
                self.backpropagate(path, value, node.state.turn_player, is_terminal=False)

        # 3. Final Selection
        return self.get_result(root, temperature)

    def backpropagate(self, path, value, turn_perspective, is_terminal):
        for node in reversed(path):
            # CRITICAL FIX: Revert VIRTUAL_LOSS from virtual_loss variable, NOT visit_count
            if not is_terminal:
                node.virtual_loss -= VIRTUAL_LOSS 
                node.value_sum += VIRTUAL_LOSS 

            # Update with real data
            node.visit_count += 1
            if node.state.turn_player == turn_perspective:
                node.value_sum += value
            else:
                node.value_sum -= value

    def add_exploration_noise(self, node):
        actions = list(node.children.keys())
        if not actions: return
        noise = np.random.dirichlet([0.3] * len(actions))
        frac = 0.25 
        for i, action in enumerate(actions):
            node.children[action].prior = node.children[action].prior * (1 - frac) + noise[i] * frac

    def get_result(self, root, temperature):
        if temperature == 0: 
            return root.best_action(), self.get_policy_vector(root)
            
        actions = list(root.children.keys())
        if not actions: return None, self.get_policy_vector(root)

        visits = np.array([root.children[a].visit_count for a in actions], dtype=np.float32)
        
        if temperature != 1.0:
            visits = np.power(visits, 1.0 / temperature)
        
        probs = visits / np.sum(visits) if np.sum(visits) > 0 else np.ones(len(visits))/len(visits)
        chosen_action = np.random.choice(actions, p=probs)
        return chosen_action, self.get_policy_vector(root)