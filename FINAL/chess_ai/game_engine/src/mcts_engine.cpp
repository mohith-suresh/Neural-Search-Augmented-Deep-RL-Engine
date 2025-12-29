#include "mcts_engine.h"
#include <cmath>
#include <algorithm>
#include <numeric>

// === select_child() - matches mcts.py Node.select_child() ===
std::pair<std::string, std::shared_ptr<MCTSNode>> MCTSNode::select_child() {
    float best_score = -1e9f;
    std::string best_action;
    std::shared_ptr<MCTSNode> best_child;
    
    float parent_visits = visit_count + virtual_loss;
    float cpuct = CPUCT_INIT + std::log((parent_visits + CPUCT_BASE) / CPUCT_BASE);
    float sqrt_parent_visits = std::sqrt(std::max(1.0f, parent_visits));
    
    for (auto& [action, child] : children) {
        float child_visits = child->visit_count + child->virtual_loss;
        float q_value = -child->value();
        float u_value = cpuct * child->prior * sqrt_parent_visits / (1.0f + child_visits);
        float score = q_value + u_value;
        
        if (score > best_score) {
            best_score = score;
            best_action = action;
            best_child = child;
        }
    }
    
    return {best_action, best_child};
}

// === expand() - matches mcts.py Node.expand() ===
void MCTSNode::expand(const std::vector<std::string>& valid_moves,
                     const std::vector<float>& policy_logits) {
    std::unordered_map<std::string, float> move_probs;
    float policy_sum = 0.0f;
    
    for (const auto& move_str : valid_moves) {
        int idx = move_to_index(move_str);
        float logit = (idx < (int)policy_logits.size()) ? policy_logits[idx] : -10.0f;
        float prob = std::exp(logit);
        move_probs[move_str] = prob;
        policy_sum += prob;
    }
    
    for (const auto& move : valid_moves) {
        float normalized_prior;
        if (policy_sum > 0) {
            normalized_prior = move_probs[move] / policy_sum;
        } else {
            normalized_prior = 1.0f / valid_moves.size();
        }
        
        py::object next_state = py_state.attr("copy")();
        next_state.attr("push")(move);
        auto child = std::make_shared<MCTSNode>(next_state, shared_from_this(), normalized_prior);
        children[move] = child;
    }
}

// === best_action() - matches mcts.py Node.best_action() ===
std::string MCTSNode::best_action() const {
    int most_visits = -1;
    std::string best = "";
    
    for (const auto& [action, child] : children) {
        if (child->visit_count > most_visits) {
            most_visits = child->visit_count;
            best = action;
        }
    }
    
    return best;
}

// === backpropagate() - matches mcts.py MCTSWorker.backpropagate() ===
void MCTSEngine::backpropagate(const std::vector<std::shared_ptr<MCTSNode>>& path,
                               float value, float leaf_turn_player) {
    // Remove virtual loss from leaf
    auto leaf = path.back();
    leaf->virtual_loss -= VIRTUAL_LOSS;
    leaf->value_sum += VIRTUAL_LOSS;
    
    // Standard backprop through all nodes
    for (auto& node : path) {
        node->visit_count += 1;
        float turn_val = py::cast<float>(node->py_state.attr("turn_player"));
        if (turn_val == leaf_turn_player) {
            node->value_sum += value;
        } else {
            node->value_sum -= value;
        }
    }
}

// === add_dirichlet_noise() - matches mcts.py add_exploration_noise() ===
void MCTSEngine::add_dirichlet_noise(std::shared_ptr<MCTSNode>& root) {
    if (root->children.empty()) return;
    
    size_t num_actions = root->children.size();
    
    // Generate Dirichlet noise using gamma distribution
    // Dirichlet(alpha) = normalize(Gamma(alpha, 1) for each dimension)
    std::gamma_distribution<float> gamma(DIRICHLET_ALPHA, 1.0f);
    
    std::vector<float> noise(num_actions);
    float noise_sum = 0.0f;
    for (size_t i = 0; i < num_actions; i++) {
        noise[i] = gamma(rng);
        noise_sum += noise[i];
    }
    
    // Normalize noise
    if (noise_sum > 0) {
        for (auto& n : noise) {
            n /= noise_sum;
        }
    }
    
    // Apply noise to priors: prior = (1 - frac) * prior + frac * noise
    size_t i = 0;
    for (auto& [action, child] : root->children) {
        child->prior = (1.0f - DIRICHLET_FRAC) * child->prior + DIRICHLET_FRAC * noise[i];
        i++;
    }
}

// === get_policy_vector() - matches mcts.py MCTSWorker.get_policy_vector() ===
py::array_t<float> MCTSEngine::get_policy_vector(const std::shared_ptr<MCTSNode>& root, float temperature) {
    std::vector<float> policy(8192, 0.0f);
    std::unordered_map<int, float> visits;
    
    for (const auto& [action_uci, child] : root->children) {
        int idx = move_to_index(action_uci);
        if (idx < 8192) {
            visits[idx] = (float)child->visit_count;
        }
    }
    
    if (visits.empty()) {
        return py::array_t<float>(8192, policy.data());
    }
    
    std::vector<float> counts;
    std::vector<int> indices;
    for (const auto& [idx, v] : visits) {
        indices.push_back(idx);
        counts.push_back(v);
    }
    
    // Apply temperature-adjusted sharpening
    float total = 0.0f;
    for (auto& c : counts) {
        float exponent = 1.3f / std::max(0.01f, temperature);
        c = std::pow(c, exponent);
        total += c;
    }
    
    if (total > 0) {
        for (size_t i = 0; i < indices.size(); i++) {
            policy[indices[i]] = counts[i] / total;
        }
    }
    
    return py::array_t<float>(8192, policy.data());
}

// ════════════════════════════════════════════════════════════════════════════════════
// MAIN SEARCH - NOW WITH CALLBACK-BASED BATCHED INFERENCE
// ════════════════════════════════════════════════════════════════════════════════════
std::pair<std::string, py::array_t<float>> MCTSEngine::search(
    py::object root_state,
    const py::array_t<float>& initial_policy,
    float initial_value,
    float temperature,
    uint32_t seed,
    py::function inference_callback) {
    
    // Seed RNG for exploration diversity
    rng.seed(seed);
    
    // Create root node
    auto root = std::make_shared<MCTSNode>(root_state);
    
    // Convert initial policy numpy array to vector
    auto policy_buf = initial_policy.request();
    std::vector<float> policy_vec((float*)policy_buf.ptr,
                                   (float*)policy_buf.ptr + policy_buf.size);
    
    // Expand root with neural network policy
    auto legal_moves = py::cast<std::vector<std::string>>(
        root_state.attr("legal_moves")());
    root->expand(legal_moves, policy_vec);
    
    // Add Dirichlet noise to root for exploration
    add_dirichlet_noise(root);
    
    // Calculate number of iterations
    int num_iterations = std::max(1, simulations / batch_size);
    
    // ════════════════════════════════════════════════════════════════════════
    // MAIN MCTS LOOP
    // ════════════════════════════════════════════════════════════════════════
    for (int iter = 0; iter < num_iterations; iter++) {
        
        std::vector<std::shared_ptr<MCTSNode>> leaves;
        std::vector<std::vector<std::shared_ptr<MCTSNode>>> paths;
        std::vector<py::object> leaf_states;  // For batched inference
        
        // ════════════════════════════════════════════════════════════════════
        // SELECTION PHASE: Traverse tree to find unexpanded leaves
        // ════════════════════════════════════════════════════════════════════
        for (int i = 0; i < batch_size; i++) {
            auto node = root;
            std::vector<std::shared_ptr<MCTSNode>> path = {node};
            
            // Traverse down the tree using PUCT selection
            while (node->is_expanded()) {
                // Epsilon-greedy exploration (5% random)
                std::uniform_real_distribution<float> epsilon_dist(0.0f, 1.0f);
                constexpr float epsilon = 0.05f;
                
                if (epsilon_dist(rng) < epsilon && node->children.size() > 1) {
                    // Random child selection
                    std::uniform_int_distribution<int> child_dist(0, node->children.size() - 1);
                    int random_idx = child_dist(rng);
                    auto it = node->children.begin();
                    std::advance(it, random_idx);
                    path.push_back(it->second);
                    node = it->second;
                } else {
                    // Normal PUCT selection
                    auto [action, next_node] = node->select_child();
                    path.push_back(next_node);
                    node = next_node;
                }
            }
            
            // Apply virtual loss to discourage other paths from selecting this node
            node->virtual_loss += VIRTUAL_LOSS;
            node->value_sum -= VIRTUAL_LOSS;
            
            // Check if terminal state
            bool is_over = py::cast<bool>(node->py_state.attr("is_over"));
            if (is_over) {
                // Terminal node - backpropagate actual game result
                float reward = py::cast<float>(
                    node->py_state.attr("get_reward_for_turn")(
                        node->py_state.attr("turn_player")));
                backpropagate(path, reward, py::cast<float>(
                    node->py_state.attr("turn_player")));
            } else {
                // Non-terminal - queue for neural network inference
                leaves.push_back(node);
                paths.push_back(path);
                leaf_states.push_back(node->py_state);
            }
        }
        
        // Skip if no leaves to expand (all terminal states)
        if (leaves.empty()) continue;
        
        // ════════════════════════════════════════════════════════════════════
        // INFERENCE PHASE: Call Python callback for batched neural network eval
        // 
        // This is the KEY FIX: Instead of using uniform policy/initial value,
        // we call back to Python which sends the batch to the GPU inference
        // server and returns actual neural network predictions.
        // ════════════════════════════════════════════════════════════════════
        
        // Convert leaf_states to Python list for callback
        py::list py_leaf_states;
        for (auto& state : leaf_states) {
            py_leaf_states.append(state);
        }
        
        // Call Python inference callback
        // Expected return: tuple(policies, values) where:
        //   policies: numpy array shape (batch_size, 8192)
        //   values: numpy array shape (batch_size,)
        py::object result = inference_callback(py_leaf_states);
        
        // Extract policies and values from result tuple
        py::array_t<float> policies_array = result.attr("__getitem__")(0).cast<py::array_t<float>>();
        py::array_t<float> values_array = result.attr("__getitem__")(1).cast<py::array_t<float>>();
        
        auto policies_buf = policies_array.request();
        auto values_buf = values_array.request();
        
        float* policies_ptr = (float*)policies_buf.ptr;
        float* values_ptr = (float*)values_buf.ptr;
        
        // ════════════════════════════════════════════════════════════════════
        // EXPANSION & BACKPROPAGATION PHASE
        // ════════════════════════════════════════════════════════════════════
        for (size_t i = 0; i < leaves.size(); i++) {
            auto node = leaves[i];
            auto& path = paths[i];
            
            // Get legal moves for this leaf
            auto next_legal = py::cast<std::vector<std::string>>(
                node->py_state.attr("legal_moves")());
            
            // Extract policy for this leaf (row i of policies array)
            // policies_array is shape (batch_size, 8192)
            std::vector<float> leaf_policy(8192);
            for (int j = 0; j < 8192; j++) {
                leaf_policy[j] = policies_ptr[i * 8192 + j];
            }
            
            // Expand with ACTUAL neural network policy
            node->expand(next_legal, leaf_policy);
            
            // Get value for this leaf
            float leaf_value = values_ptr[i];
            
            // Backpropagate ACTUAL neural network value
            backpropagate(path, leaf_value, py::cast<float>(
                node->py_state.attr("turn_player")));
        }
    }
    
    // ════════════════════════════════════════════════════════════════════════
    // RETURN RESULT
    // ════════════════════════════════════════════════════════════════════════
    std::string best_move = root->best_action();
    auto policy = get_policy_vector(root, temperature);
    
    return {best_move, policy};
}
