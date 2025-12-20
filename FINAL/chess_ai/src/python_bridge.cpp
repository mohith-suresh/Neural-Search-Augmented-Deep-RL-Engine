#include "mcts_engine.h"

// === pybind11 Python Module Bindings ===
// Exposes MCTSEngine class to Python so it can be called as:
//   from mcts_engine_cpp import MCTSEngine
//   engine = MCTSEngine(simulations=800, batch_size=8)
//   best_move, policy = engine.search(root_state, policy, value, temperature)

PYBIND11_MODULE(mcts_engine_cpp, m) {
    m.doc() = "MCTS C++ Engine - Fast tree traversal with Python callbacks";
    
    py::class_<MCTSEngine>(m, "MCTSEngine")
        .def(py::init<int, int>(), 
             py::arg("simulations") = 800, 
             py::arg("batch_size") = 8,
             "Initialize MCTS engine with simulation count and batch size")
        
        .def("search", &MCTSEngine::search,
             py::arg("root_state"),
             py::arg("initial_policy"),
             py::arg("initial_value"),
             py::arg("temperature") = 1.0f,
             "Perform MCTS search from root_state with given policy and value")
        
        .def_readwrite("simulations", &MCTSEngine::simulations, "Number of simulations")
        .def_readwrite("batch_size", &MCTSEngine::batch_size, "Batch size for parallel search");
}