#pragma once

#include <vector>
#include <unordered_map>
#include <memory>
#include <string>
#include <cmath>
#include <algorithm>
#include <random>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// === CONSTANTS (from mcts.py) ===
const float VIRTUAL_LOSS = 3.0f;      // mcts.py: VIRTUAL_LOSS = 3.0
const float CPUCT_INIT = 1.0f;        // mcts.py: CPUCT_INIT = 1.0
const int CPUCT_BASE = 19652;         // mcts.py: CPUCT_BASE = 19652

// === Move Encoding (matches mcts.py move_to_index exactly) ===
inline int move_to_index(const std::string& move_str) {
    // Python: src = (ord(move_str[0]) - 97) + (ord(move_str[1]) - 49) * 8
    int src = (move_str[0] - 'a') + (move_str[1] - '1') * 8;
    // Python: dst = (ord(move_str[2]) - 97) + (ord(move_str[3]) - 49) * 8
    int dst = (move_str[2] - 'a') + (move_str[3] - '1') * 8;
    // Python: idx = src * 64 + dst
    int idx = src * 64 + dst;
    
    // Python: if len(move_str) == 5 and move_str[4] != 'q':
    if (move_str.length() == 5 && move_str[4] != 'q') {
        int promo_type = 0;
        // Python: type_map = {'n': 0, 'b': 1, 'r': 2}
        if (move_str[4] == 'n') promo_type = 0;
        else if (move_str[4] == 'b') promo_type = 1;
        else if (move_str[4] == 'r') promo_type = 2;
        
        // Python: src_col = src % 8; dst_col = dst % 8; direction = (dst_col - src_col) + 1
        int src_col = src % 8;
        int dst_col = dst % 8;
        int direction = (dst_col - src_col) + 1;
        // Python: idx = 4096 + (src_col * 9) + (direction * 3) + promo_type
        idx = 4096 + (src_col * 9) + (direction * 3) + promo_type;
    }
    return (idx < 8192) ? idx : 0;
}

// === MCTSNode Class (matches mcts.py Node class) ===
class MCTSNode : public std::enable_shared_from_this<MCTSNode> {
public:
    py::object py_state;  // Python: self.state = state
    std::unordered_map<std::string, std::shared_ptr<MCTSNode>> children;  // Python: self.children = {}
    std::shared_ptr<MCTSNode> parent;  // Python: self.parent = parent (implicit)
    
    // Python: self.visit_count = 0
    int visit_count = 0;
    // Python: self.value_sum = 0
    float value_sum = 0.0f;
    // Python: self.prior = prior
    float prior = 0.0f;
    // Python: self.virtual_loss = 0
    float virtual_loss = 0.0f;
    
    MCTSNode() = default;
    MCTSNode(py::object state, std::shared_ptr<MCTSNode> p = nullptr, float pr = 0.0f)
        : py_state(state), parent(p), prior(pr) {}
    
    // Delete copy (mutex not needed, just smart pointer management)
    MCTSNode(const MCTSNode&) = delete;
    MCTSNode& operator=(const MCTSNode&) = delete;
    MCTSNode(MCTSNode&&) = default;
    MCTSNode& operator=(MCTSNode&&) = default;
    
    // Python: def is_expanded(self): return len(self.children) > 0
    bool is_expanded() const { 
        return !children.empty(); 
    }
    
    // Python: def value(self):
    //     visits = self.visit_count + self.virtual_loss
    //     if visits <= 0: return 0
    //     return self.value_sum / visits
    float value() const {
        float visits = visit_count + virtual_loss;
        return (visits > 0) ? value_sum / visits : 0.0f;
    }
    
    // Python: def select_child(self): ... returns (best_action, best_child)
    std::pair<std::string, std::shared_ptr<MCTSNode>> select_child();
    
    // Python: def expand(self, valid_moves, policy_logits): ...
    void expand(const std::vector<std::string>& valid_moves, 
                const std::vector<float>& policy_logits);
    
    // Python: def best_action(self): ... returns best_action_str
    std::string best_action() const;
};

// === MCTSEngine Class (wraps search algorithm) ===
class MCTSEngine {
public:
    int simulations;  // Number of MCTS simulations
    int batch_size;   // Batch size for parallel search
    std::mt19937 rng;
    
    MCTSEngine(int sims = 800, int bs = 8) : simulations(sims), batch_size(bs) {}
    
    // Python: def search(self, root_state, temperature=1.0): ...
    // Returns: (best_move, policy_vector)
    std::pair<std::string, py::array_t<float>> search(
        py::object root_state,
        const py::array_t<float>& initial_policy,
        float initial_value,
        float temperature,
        uint32_t seed = 0 
    );

    
private:
    // Python: def backpropagate(self, path, value, leaf_turn_player, is_terminal): ...
    void backpropagate(const std::vector<std::shared_ptr<MCTSNode>>& path,
                      float value, float leaf_turn_player);
    
    // Python: def get_policy_vector(self, root, alpha=1.3): ...
    py::array_t<float> get_policy_vector(
    const std::shared_ptr<MCTSNode>& root,
    float temperature = 1.0f);

};