import torch
import math
import copy
import numpy as np

# --- Constants ---
VIRTUAL_LOSS = 3.0  

def move_to_index(move_str):
    src = (ord(move_str[0]) - 97) + (ord(move_str[1]) - 49) * 8
    dst = (ord(move_str[2]) - 97) + (ord(move_str[3]) - 49) * 8
    idx = src * 64 + dst
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
        self.virtual_loss = 0 
        
    def is_expanded(self):
        return len(self.children) > 0

    def value(self):
        visits = self.visit_count + self.virtual_loss
        if visits <= 0: return 0
        return self.value_sum / visits

    def select_child(self, cpuct):
        best_score = -float('inf')
        best_action = None
        best_child = None
        
        parent_visits = self.visit_count + self.virtual_loss
        sqrt_parent_visits = math.sqrt(max(1, parent_visits))
        
        for action, child in self.children.items():
            child_visits = child.visit_count + child.virtual_loss
            q_value = child.value()
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
        
        # Bootstrap Root
        tensor = torch.from_numpy(root.state.to_tensor())
        self.input_queue.put((self.worker_id, tensor)) 
        policy, value = self.output_queue.get()
        root.expand(root.state.legal_moves(), policy)
        self.add_exploration_noise(root)
        
        # Batch Loop
        num_iterations = max(1, self.simulations // self.batch_size)
        
        for _ in range(num_iterations):
            leaves = []
            paths = []
            tensors = []
            
            for _ in range(self.batch_size):
                node = root
                path = [node]
                while node.is_expanded():
                    action, node = node.select_child(self.cpu)
                    path.append(node)
                    node.virtual_loss += VIRTUAL_LOSS
                    node.value_sum -= VIRTUAL_LOSS 
                
                if node.state.is_over:
                    reward = node.state.get_reward_for_turn(node.state.turn_player)
                    self.backpropagate(path, reward, node.state.turn_player, is_terminal=True)
                else:
                    leaves.append(node)
                    paths.append(path)
                    tensors.append(node.state.to_tensor())

            if not leaves: continue

            # Inference
            batch_tensor = torch.from_numpy(np.array(tensors))
            self.input_queue.put((self.worker_id, batch_tensor))
            policies, values = self.output_queue.get()
            
            for i, node in enumerate(leaves):
                path = paths[i]
                node.expand(node.state.legal_moves(), policies[i])
                self.backpropagate(path, values[i], node.state.turn_player, is_terminal=False)

        return self.get_result(root, temperature)

    def backpropagate(self, path, value, turn_perspective, is_terminal):
        for node in reversed(path):
            if not is_terminal:
                node.virtual_loss -= VIRTUAL_LOSS 
                node.value_sum += VIRTUAL_LOSS 
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
        if temperature != 1.0: visits = np.power(visits, 1.0 / temperature)
        probs = visits / np.sum(visits) if np.sum(visits) > 0 else np.ones(len(visits))/len(visits)
        return np.random.choice(actions, p=probs), self.get_policy_vector(root)