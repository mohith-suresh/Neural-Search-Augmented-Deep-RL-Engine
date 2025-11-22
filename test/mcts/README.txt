MCTS + Self-Play Implementation for Chess AI
=============================================

Project: EE542 - Deconstructing AlphaZero's Success
Directory: test/mcts/

This directory contains a complete implementation of Monte Carlo Tree Search (MCTS)
with neural network guidance, self-play game generation, and comprehensive evaluation.

FILES
-----

1. mcts_tree.py
   - Core MCTS implementation with UCB exploration
   - MCTSNode: Tree node with visit counts, values, priors
   - MCTS: Search algorithm with neural network guidance
   - Features: Dirichlet noise, temperature control, virtual loss

2. self_play.py
   - Self-play game generation for training data
   - SelfPlayGame: Single game with MCTS move selection
   - SelfPlayWorker: Batch game generation
   - Output: (position, policy, value) training examples

3. neural_mcts_player.py
   - Integration of trained CNN with MCTS
   - NeuralNetWrapper: Adapts PyTorch CNN to MCTS interface
   - NeuralMCTSPlayer: Complete chess player
   - Features: Move selection, position evaluation, game playing

4. mcts_evaluation.py
   - Comprehensive MCTS evaluation framework
   - Tests: Search quality, efficiency, playing strength
   - Parameter sensitivity analysis
   - Report generation with metrics

5. run_full_evaluation.py
   - Complete evaluation pipeline
   - Runs all evaluation stages automatically
   - Generates comprehensive report
   - ELO estimation with Stockfish

USAGE
-----

Step 1: Train a CNN model (required first)
-------------------------------------------
cd ../../FINAL/chess_ai/game_engine
python cnn.py

This will create: model/best_model.pth


Step 2: Test MCTS with trained model
-------------------------------------
cd ../../../test/mcts
python neural_mcts_player.py

This will:
- Load the trained model
- Evaluate starting position
- Play a test game vs random opponent


Step 3: Generate self-play games
---------------------------------
python self_play.py

This will:
- Generate 10 test self-play games
- Save training examples to selfplay_games/
- Each game produces ~40-80 training examples


Step 4: Run complete evaluation
--------------------------------
python run_full_evaluation.py --model /path/to/best_model.pth

Full evaluation (takes ~1-2 hours):
python run_full_evaluation.py --model ../../FINAL/chess_ai/game_engine/model/best_model.pth

Quick evaluation (takes ~10-15 minutes):
python run_full_evaluation.py --model ../../FINAL/chess_ai/game_engine/model/best_model.pth --quick

This will:
- Load model and test basic functionality
- Evaluate MCTS search quality (50 positions)
- Test playing strength vs random (30 games)
- Analyze parameter sensitivity
- Generate self-play games (20 games)
- Estimate ELO with Stockfish
- Generate comprehensive report


Step 5: View results
--------------------
Results will be saved to: mcts_evaluation_results/

Files:
- evaluation_report.txt: Human-readable report
- results.json: Structured JSON results
- evaluation.log: Detailed execution log
- selfplay_games/: Generated training data


ALGORITHM OVERVIEW
------------------

MCTS Search (mcts_tree.py):
1. Selection: Traverse tree using UCB until leaf node
   UCB = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
   where:
   - Q(s,a): Mean action value (exploitation)
   - P(s,a): Neural network prior (policy)
   - c_puct: Exploration constant (typically 1.0)

2. Expansion: Create children for all legal moves
   Priors set from neural network policy output

3. Evaluation: Neural network forward pass
   Returns (policy_probs, value)

4. Backpropagation: Update statistics up tree
   Values negated at each level (zero-sum game)


Self-Play (self_play.py):
1. Start from initial position
2. For each move:
   a. Run MCTS search (800 simulations)
   b. Add Dirichlet noise for exploration (early game)
   c. Select move from visit distribution
   d. Store (position, policy, value) example
3. Play until terminal or max length
4. Label all positions with final outcome


Evaluation (mcts_evaluation.py):
1. Search Quality:
   - Move ranking correlation
   - Value prediction accuracy
   - Search depth analysis

2. Computational Efficiency:
   - Nodes per second
   - Time per simulation
   - Scaling with simulation count

3. Playing Strength:
   - Win rate vs baselines
   - ELO estimation
   - Tactical test positions

4. Parameter Sensitivity:
   - c_puct values
   - Simulation counts
   - Temperature effects


PARAMETERS
----------

MCTS Parameters (mcts_tree.py):
- num_simulations: 800 (AlphaZero standard)
  Trade-off: More sims = stronger play but slower
  Tested: 50, 100, 200, 400, 800

- c_puct: 1.0 (exploration constant)
  Trade-off: Higher = more exploration
  Tested: 0.5, 1.0, 1.5, 2.0

- temperature: 0.0 to 1.0
  0.0 = Deterministic (argmax)
  1.0 = Proportional to visits
  Schedule: 1.0 for first 30 moves, then 0.1

- dirichlet_alpha: 0.3 (chess standard)
  Controls exploration noise strength

- dirichlet_epsilon: 0.25
  Mixing weight for noise


Self-Play Parameters (self_play.py):
- temp_threshold: 30 moves
  Switch from exploration to exploitation

- max_game_length: 500 moves
  Prevent infinite games

- resign_threshold: -0.9
  Value threshold for resignation (disabled by default)


MEMORY REQUIREMENTS
-------------------

MCTS Tree:
- 800 simulations: ~5-10 MB per position
- Nodes store: visit_count, total_value, prior_prob, children

Self-Play:
- 1 game (~50 moves): ~15 MB
- 100 games: ~1.5 GB
- Uses chunked saving to manage memory

Model Inference:
- Batch size 1: ~200 MB GPU memory
- Full model: ~500 MB GPU memory


PERFORMANCE BENCHMARKS
----------------------

Expected performance (RTX 3060):

MCTS Search:
- 100 simulations: ~0.5s per move
- 400 simulations: ~2.0s per move
- 800 simulations: ~4.0s per move

Self-Play Generation:
- 1 game (100 sims/move): ~3-5 minutes
- 1 game (400 sims/move): ~10-15 minutes
- 20 games (200 sims): ~2-3 hours

Evaluation:
- Quick mode: ~10-15 minutes
- Full mode: ~1-2 hours


EXPECTED ELO
------------

Based on training data and architecture:

Supervised CNN only (no MCTS):
- Random play: ~1200-1400 ELO
- Greedy (best first move): ~1400-1600 ELO

Supervised CNN + MCTS (100 sims):
- Expected: ~1600-1800 ELO

Supervised CNN + MCTS (400 sims):
- Expected: ~1800-2000 ELO

Supervised CNN + MCTS + Self-Play:
- Expected: ~2000-2200 ELO


TROUBLESHOOTING
---------------

Issue: "Model not found"
Solution: Train model first using cnn.py in FINAL/chess_ai/game_engine/

Issue: "CUDA out of memory"
Solution: Reduce batch size or use CPU mode (device="cpu")

Issue: "Stockfish not available"
Solution: Install stockfish: sudo apt install stockfish

Issue: MCTS very slow
Solution: Reduce num_simulations (try 100 or 200)

Issue: Self-play games too short
Solution: Check resign_enabled=False in config

Issue: Import errors
Solution: Ensure you're running from test/mcts/ directory


INTEGRATION WITH TRAINING
--------------------------

To use self-play data for training:

1. Generate self-play games:
   python self_play.py

2. Convert to training format:
   Self-play output is already in correct format:
   - positions: (N, 12, 8, 8) float32
   - policies: (N, 8192) float32 (MCTS visit distributions)
   - values: (N,) float32 (game outcomes)

3. Combine with supervised data:
   # Mix 90% supervised + 10% self-play
   supervised = np.load('classical_20M.npz')
   selfplay = np.load('selfplay_games_20.npz')

   combined_positions = np.concatenate([supervised['positions'], selfplay['positions']])
   combined_moves = np.concatenate([supervised['moves'], selfplay['policies']])
   combined_results = np.concatenate([supervised['results'], selfplay['values']])

4. Retrain model:
   python cnn.py --data combined_data.npz


REFERENCES
----------

[1] Silver et al. (2017): "Mastering Chess and Shogi by Self-Play with a
    General Reinforcement Learning Algorithm" (AlphaZero)
    https://arxiv.org/abs/1712.01815

[2] Browne et al. (2012): "A Survey of Monte Carlo Tree Search Methods"
    IEEE Transactions on Computational Intelligence and AI in Games

[3] Coulom (2006): "Efficient Selectivity and Backup Operators in
    Monte-Carlo Tree Search"

[4] Kocsis & Szepesvari (2006): "Bandit based Monte-Carlo Planning"
    (UCT algorithm)


PROJECT ALIGNMENT
-----------------

This implementation supports the research goals from the project proposal:

Hypothesis Testing:
1. CNN spatial understanding (70-80% of strength)
   -> Test by comparing CNN-only vs CNN+MCTS

2. Optimized MCTS (50-200 vs 80,000 simulations)
   -> Parameter sensitivity analysis shows performance vs simulation count

3. Self-play RL contribution (10-20% marginal)
   -> Compare supervised baseline vs supervised+selfplay

Evaluation Metrics:
- ELO estimation (Stockfish benchmark)
- Win rates vs baselines
- Computational efficiency (GPU hours)
- Training data quality (self-play examples)


NEXT STEPS
----------

After completing evaluation:

1. Analyze results to test hypothesis
2. Compare supervised-only vs supervised+MCTS vs supervised+MCTS+selfplay
3. Generate ablation studies
4. Prepare visualizations for presentation
5. Document findings in final report


CONTACT
-------

For issues or questions about this implementation:
- Check evaluation logs in mcts_evaluation_results/evaluation.log
- Review error messages in console output
- Verify model path and configuration parameters
