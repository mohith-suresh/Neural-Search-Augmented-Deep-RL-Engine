import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import send_from_directory, Flask, request, jsonify
from flask_socketio import SocketIO, emit
from game_engine.chess_env import ChessGame
from config.settings import get_config
import threading
import time

app = Flask(__name__)
app.config.from_object(get_config())
socketio = SocketIO(app, cors_allowed_origins=app.config['CORS_ORIGINS'])

games = {}  # game_id: ChessGame instance
auto_play_threads = {}  # game_id: thread control

def get_game(game_id):
    return games.get(game_id, None)

@app.route("/")
def index():
    frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
    return send_from_directory(frontend_dir, "index.html")

@app.route("/static/<path:filename>")
def serve_static(filename):
    static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend", "static"))
    return send_from_directory(static_dir, filename)

@app.route("/new_game", methods=["POST"])
def new_game():
    game = ChessGame()
    games[game.game_id] = game
    return jsonify({"game_id": game.game_id, **game.get_state()})

@app.route("/game/<game_id>", methods=["GET"])
def get_game_state(game_id):
    game = get_game(game_id)
    if not game:
        return jsonify({"error": "No such game_id"}), 404
    return jsonify(game.get_state())

@app.route("/move/<game_id>", methods=["POST"])
def make_move(game_id):
    game = get_game(game_id)
    if not game:
        return jsonify({"error": "No such game_id"}), 404
    req = request.get_json() or {}
    move = req.get("move")
    if not move:
        return jsonify({"error": "Missing 'move' field"}), 400
    success = game.push(move)
    return jsonify({"success": success, **game.get_state()})

@app.route("/random_move/<game_id>", methods=["POST"])
def random_move(game_id):
    game = get_game(game_id)
    if not game:
        return jsonify({"error": "No such game_id"}), 404
    move = game.random_move()
    return jsonify({"move": move, **game.get_state()})

@app.route("/reset/<game_id>", methods=["POST"])
def reset_game(game_id):
    game = get_game(game_id)
    if not game:
        return jsonify({"error": "No such game_id"}), 404
    game.reset()
    return jsonify(game.get_state())

@app.route("/game_stats/<game_id>", methods=["GET"])
def game_stats(game_id):
    game = get_game(game_id)
    if not game:
        return jsonify({"error": "No such game_id"}), 404
    
    stats = {
        "total_moves": len(game.moves),
        "move_history": game.moves,
        "is_over": game.is_over,
        "result": game.result,
        "current_turn": "White" if game.board.turn else "Black",
        "legal_moves_count": len(list(game.board.legal_moves))
    }
    return jsonify(stats)

# SocketIO events for real-time features
@socketio.on('new_game')
def on_new_game():
    game = ChessGame()
    games[game.game_id] = game
    emit('game_state', {"game_id": game.game_id, **game.get_state()})

@socketio.on('make_move')
def on_make_move(data):
    game_id = data.get('game_id')
    move = data.get('move')
    game = get_game(game_id)
    if not game:
        emit('error', {'error': 'No such game_id'})
        return
    success = game.push(move)
    emit('game_state', {"success": success, **game.get_state()})

@socketio.on('start_auto_play')
def on_start_auto_play(data):
    game_id = data.get('game_id')
    delay = data.get('delay', 0.5)
    
    game = get_game(game_id)
    if not game:
        emit('error', {'error': 'No such game_id'})
        return
    
    def auto_play():
        while game_id in auto_play_threads and not game.is_over:
            move = game.random_move()
            if move:
                socketio.emit('game_state', {
                    "game_id": game_id,
                    **game.get_state(),
                    "auto_play": True
                })
                time.sleep(delay)
            else:
                break
        
        if game_id in auto_play_threads:
            del auto_play_threads[game_id]
    
    if game_id not in auto_play_threads:
        thread = threading.Thread(target=auto_play)
        thread.daemon = True
        auto_play_threads[game_id] = thread
        thread.start()
        emit('message', {'text': 'Auto-play started'})

@socketio.on('stop_auto_play')
def on_stop_auto_play(data):
    game_id = data.get('game_id')
    if game_id in auto_play_threads:
        del auto_play_threads[game_id]
        emit('message', {'text': 'Auto-play stopped'})

@socketio.on('reset')
def on_reset(data):
    game_id = data.get('game_id')
    game = get_game(game_id)
    if not game:
        emit('error', {'error': 'No such game_id'})
        return
    game.reset()
    emit('game_state', game.get_state())

if __name__ == "__main__":
    socketio.run(app, host=app.config['HOST'], port=app.config['PORT'], debug=app.config['DEBUG'])
