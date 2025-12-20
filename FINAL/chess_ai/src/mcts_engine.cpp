#include "mcts_engine.h"
#include <cmath>
#include <algorithm>

// === select_child() - matches mcts.py Node.select_child() ===
std::pair<std::string, std::shared_ptr<MCTSNode>> MCTSNode::select_child() {
    // Python:
    //     best_score = -float('inf')
    //     best_action = None
    //     best_child = None
    float best_score = -1e9f;
    std::string best_action;
    std::shared_ptr<MCTSNode> best_child;
    
    // Python: parent_visits = self.visit_count + self.virtual_loss
    float parent_visits = visit_count + virtual_loss;
    
    // Python: cpuct = CPUCT_INIT + math.log((parent_visits + CPUCT_BASE) / CPUCT_BASE)
    float cpuct = CPUCT_INIT + std::log((parent_visits + CPUCT_BASE) / CPUCT_BASE);
    
    // Python: sqrt_parent_visits = math.sqrt(max(1, parent_visits))
    float sqrt_parent_visits = std::sqrt(std::max(1.0f, parent_visits));
    
    // Python: for action, child in self.children.items():
    for (auto& [action, child] : children) {
        // Python: child_visits = child.visit_count + child.virtual_loss
        float child_visits = child->visit_count + child->virtual_loss;
        
        // Python: q_value = -child.value()
        float q_value = -child->value();
        
        // Python: u_value = cpuct * child.prior * sqrt_parent_visits / (1 + child_visits)
        float u_value = cpuct * child->prior * sqrt_parent_visits / (1.0f + child_visits);
        
        // Python: score = q_value + u_value
        float score = q_value + u_value;
        
        // Python: if score > best_score: ...
        if (score > best_score) {
            best_score = score;
            best_action = action;
            best_child = child;
        }
    }
    
    // Python: return best_action, best_child
    return {best_action, best_child};
}

// === expand() - matches mcts.py Node.expand() ===
void MCTSNode::expand(const std::vector<std::string>& valid_moves,
                     const std::vector<float>& policy_logits) {
    // Python: move_probs = {}; policy_sum = 0
    std::unordered_map<std::string, float> move_probs;
    float policy_sum = 0.0f;
    
    // Python: for move_str in valid_moves:
    for (const auto& move_str : valid_moves) {
        // Python: idx = move_to_index(move_str)
        int idx = move_to_index(move_str);
        
        // Python: logit = policy_logits[idx] if idx < len(policy_logits) else -10.0
        float logit = (idx < (int)policy_logits.size()) ? policy_logits[idx] : -10.0f;
        
        // Python: prob = math.exp(logit)
        float prob = std::exp(logit);
        
        // Python: move_probs[move_str] = prob; policy_sum += prob
        move_probs[move_str] = prob;
        policy_sum += prob;
    }
    
    // Python: for move in valid_moves:
    for (const auto& move : valid_moves) {
        // Python: if policy_sum > 0:
        //            normalized_prior = move_probs[move] / policy_sum
        //         else:
        //            normalized_prior = 1.0 / len(valid_moves)
        float normalized_prior;
        if (policy_sum > 0) {
            normalized_prior = move_probs[move] / policy_sum;
        } else {
            normalized_prior = 1.0f / valid_moves.size();
        }
        
        // Python: next_state = self.state.copy()
        py::object next_state = py_state.attr("copy")();
        
        // Python: next_state.push(move)
        next_state.attr("push")(move);
        
        // Python: self.children[move] = Node(next_state, parent=self, prior=normalized_prior)
        auto child = std::make_shared<MCTSNode>(next_state, shared_from_this(), normalized_prior);
        children[move] = child;
    }
}

// === best_action() - matches mcts.py Node.best_action() ===
std::string MCTSNode::best_action() const {
    // Python: most_visits = -1; best_action = None
    int most_visits = -1;
    std::string best = "";
    
    // Python: for action, child in self.children.items():
    for (const auto& [action, child] : children) {
        // Python: if child.visit_count > most_visits:
        if (child->visit_count > most_visits) {
            // Python: most_visits = child.visit_count; best_action = action
            most_visits = child->visit_count;
            best = action;
        }
    }
    
    // Python: return best_action
    return best;
}

// === backpropagate() - matches mcts.py MCTSWorker.backpropagate() ===
void MCTSEngine::backpropagate(const std::vector<std::shared_ptr<MCTSNode>>& path,
                               float value, float leaf_turn_player) {
    // Python: leaf = path[-1]
    auto leaf = path.back();
    
    // Python: leaf.virtual_loss -= VIRTUAL_LOSS
    leaf->virtual_loss -= VIRTUAL_LOSS;
    
    // Python: leaf.value_sum += VIRTUAL_LOSS
    leaf->value_sum += VIRTUAL_LOSS;
    
    // Python: for node in reversed(path):
    for (auto node : path) {
        // Python: node.visit_count += 1
        node->visit_count += 1;
        
        // Python: if node.state.turn_player == leaf_turn_player:
        float turn_val = py::cast<float>(node->py_state.attr("turn_player"));
        if (turn_val == leaf_turn_player) {
            // Python: node.value_sum += value
            node->value_sum += value;
        } else {
            // Python: node.value_sum -= value
            node->value_sum -= value;
        }
    }
}

// === get_policy_vector() - matches mcts.py MCTSWorker.get_policy_vector() ===
py::array_t<float> MCTSEngine::get_policy_vector(const std::shared_ptr<MCTSNode>& root) {
    // Python: policy_vector = np.zeros(8192, dtype=np.float32)
    std::vector<float> policy(8192, 0.0f);
    
    // Python: visits = {}
    std::unordered_map<int, float> visits;
    
    // Python: for action_uci, child in root.children.items():
    for (const auto& [action_uci, child] : root->children) {
        // Python: idx = move_to_index(action_uci)
        int idx = move_to_index(action_uci);
        
        // Python: if idx < 8192: visits[idx] = child.visit_count
        if (idx < 8192) {
            visits[idx] = (float)child->visit_count;
        }
    }
    
    // Python: if not visits: return policy_vector
    if (visits.empty()) {
        return py::array_t<float>(8192, policy.data());
    }
    
    // Python: counts = np.array(list(visits.values()), dtype=np.float32)
    // Python: indices = np.array(list(visits.keys()), dtype=np.int32)
    std::vector<float> counts;
    std::vector<int> indices;
    for (const auto& [idx, v] : visits) {
        indices.push_back(idx);
        counts.push_back(v);
    }
    
    // Python: sharpened = counts ** alpha (alpha = 1.3)
    float total = 0.0f;
    for (auto& c : counts) {
        c = std::pow(c, 1.3f);
        total += c;
    }
    
    // Python: if total > 0: probs = sharpened / total; policy_vector[indices] = probs
    if (total > 0) {
        for (size_t i = 0; i < indices.size(); i++) {
            policy[indices[i]] = counts[i] / total;
        }
    }
    
    // Return numpy array (Python will convert back)
    return py::array_t<float>(8192, policy.data());
}

// === search() - main MCTS algorithm (matches mcts.py MCTSWorker.search()) ===
std::pair<std::string, py::array_t<float>> MCTSEngine::search(
    py::object root_state,
    const py::array_t<float>& initial_policy,
    float initial_value,
    float temperature) {
    
    // Python: root = Node(root_state)
    auto root = std::make_shared<MCTSNode>(root_state);
    
    // Convert numpy array to vector
    auto policy_buf = initial_policy.request();
    std::vector<float> policy_vec((float*)policy_buf.ptr, 
                                  (float*)policy_buf.ptr + policy_buf.size);
    
    // Python: root.expand(root.state.legal_moves(), policy)
    auto legal_moves = py::cast<std::vector<std::string>>(
        root_state.attr("legal_moves")());
    root->expand(legal_moves, policy_vec);
    
    // Python: num_iterations = max(1, self.simulations // self.batch_size)
    int num_iterations = std::max(1, simulations / batch_size);
    
    // Python: for _ in range(num_iterations):
    for (int iter = 0; iter < num_iterations; iter++) {
        // Python: leaves = []; paths = []; tensors = []
        std::vector<std::shared_ptr<MCTSNode>> leaves;
        std::vector<std::vector<std::shared_ptr<MCTSNode>>> paths;
        
        // Python: for _ in range(self.batch_size):
        for (int i = 0; i < batch_size; i++) {
            // Python: node = root; path = [node]
            auto node = root;
            std::vector<std::shared_ptr<MCTSNode>> path = {node};
            
            // Python: while node.is_expanded(): action, node = node.select_child(); path.append(node)
            while (node->is_expanded()) {
                auto [action, next_node] = node->select_child();
                path.push_back(next_node);
                node = next_node;
            }
            
            // Python: node.virtual_loss += VIRTUAL_LOSS
            node->virtual_loss += VIRTUAL_LOSS;
            
            // Python: node.value_sum -= VIRTUAL_LOSS
            node->value_sum -= VIRTUAL_LOSS;
            
            // Python: if node.state.is_over:
            bool is_over = py::cast<bool>(node->py_state.attr("is_over"));
            if (is_over) {
                // Python: reward = node.state.get_reward_for_turn(node.state.turn_player)
                float reward = py::cast<float>(
                    node->py_state.attr("get_reward_for_turn")(
                        node->py_state.attr("turn_player")));
                
                // Python: self.backpropagate(path, reward, node.state.turn_player, is_terminal=True)
                backpropagate(path, reward, py::cast<float>(
                    node->py_state.attr("turn_player")));
            } else {
                // Python: leaves.append(node); paths.append(path)
                leaves.push_back(node);
                paths.push_back(path);
            }
        }
        
        // Python: if not leaves: continue
        if (leaves.empty()) continue;
        
        // Expansion Phase
        // Python: for i, node in enumerate(leaves):
        for (size_t i = 0; i < leaves.size(); i++) {
            auto node = leaves[i];
            auto path = paths[i];
            
            // Python: node.expand(node.state.legal_moves(), policies[i])
            auto next_legal = py::cast<std::vector<std::string>>(
                node->py_state.attr("legal_moves")());
            // Using uniform policy (would be replaced with actual neural net policy in production)
            std::vector<float> uniform_policy(8192, 0.0f);
            node->expand(next_legal, uniform_policy);
            
            // Python: self.backpropagate(path, values[i], node.state.turn_player, is_terminal=False)
            backpropagate(path, initial_value, py::cast<float>(
                node->py_state.attr("turn_player")));
        }
    }
    
    // Python: return self.get_result(root, temperature)
    // For now, just return best action and policy vector
    std::string best_move = root->best_action();
    auto policy = get_policy_vector(root);
    
    return {best_move, policy};
}