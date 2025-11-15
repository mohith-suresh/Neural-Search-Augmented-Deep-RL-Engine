import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import send_from_directory, Flask, request, jsonify
from flask_socketio import SocketIO, emit
from game_engine.chess_env import ChessGame
from config.settings import get_config
import threading
import time
import json
from datetime import datetime

app = Flask(__name__)
app.config.from_object(get_config())
socketio = SocketIO(app, cors_allowed_origins=app.config['CORS_ORIGINS'])

games = {}
auto_play_threads = {}

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
    data = request.get_json() or {}
    mode = data.get('mode', 'pvp')
    ai_side = data.get('ai_side', 'black')
    
    game = ChessGame()
    games[game.game_id] = {
        'game': game,
        'mode': mode,
        'ai_side': ai_side,
        'timer': {'white': 600, 'black': 600},
        'created_at': datetime.now().isoformat()
    }
    
    return jsonify({
        "game_id": game.game_id,
        "mode": mode,
        "ai_side": ai_side,
        **game.get_state()
    })

@app.route("/game_stats/<game_id>", methods=["GET"])
def game_stats(game_id):
    game_data = get_game(game_id)
    if not game_data:
        return jsonify({"error": "No such game_id"}), 404
    
    game = game_data['game']
    stats = {
        "total_moves": len(game.moves),
        "move_history": game.moves,
        "is_over": game.is_over,
        "result": game.result,
        "current_turn": "White" if game.board.turn else "Black",
        "legal_moves_count": len(list(game.board.legal_moves)),
        "mode": game_data['mode'],
        "timer": game_data['timer']
    }
    return jsonify(stats)

@app.route("/save_game/<game_id>", methods=["POST"])
def save_game(game_id):
    game_data = get_game(game_id)
    if not game_data:
        return jsonify({"error": "No such game_id"}), 404
    
    game = game_data['game']
    os.makedirs('data/played_games', exist_ok=True)
    
    filename = f"data/played_games/game_{game_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump({
            'game_id': game_id,
            'mode': game_data['mode'],
            'moves': game.moves,
            'result': game.result,
            'move_count': len(game.moves),
            'created_at': game_data['created_at'],
            'finished_at': datetime.now().isoformat()
        }, f, indent=2)
    
    return jsonify({"success": True, "filename": filename})

@socketio.on('new_game')
def on_new_game(data):
    mode = data.get('mode', 'pvp')
    ai_side = data.get('ai_side', 'black')
    
    game = ChessGame()
    games[game.game_id] = {
        'game': game,
        'mode': mode,
        'ai_side': ai_side,
        'timer': {'white': 600, 'black': 600},
        'created_at': datetime.now().isoformat()
    }
    
    emit('game_created', {
        'game_id': game.game_id,
        'mode': mode,
        'ai_side': ai_side,
        **game.get_state()
    })
    
    if mode == 'pve' and ai_side == 'white':
        time.sleep(0.5)
        ai_make_move(game.game_id)

@socketio.on('player_move')
def on_player_move(data):
    game_id = data.get('game_id')
    move = data.get('move')
    
    game_data = get_game(game_id)
    if not game_data:
        emit('error', {'message': 'Game not found'})
        return
    
    game = game_data['game']
    
    if move not in game.legal_moves():
        emit('illegal_move', {'move': move})
        return
    
    success = game.push(move)
    
    if success:
        game_ending_info = check_game_ending(game)
        
        emit('move_made', {
            'move': move,
            **game.get_state(),
            **game_ending_info
        }, broadcast=True)
        
        if game.is_over:
            emit('game_over', {
                'result': game.result,
                **game_ending_info,
                **game.get_state()
            }, broadcast=True)
        elif game_data['mode'] == 'pve' and not game.is_over:
            emit('ai_thinking', {})
            socketio.start_background_task(ai_make_move_async, game_id)

@socketio.on('undo_move')
def on_undo_move(data):
    game_id = data.get('game_id')
    game_data = get_game(game_id)
    
    if not game_data:
        emit('error', {'message': 'Game not found'})
        return
    
    game = game_data['game']
    
    if len(game.moves) > 0:
        game.board.pop()
        game.moves.pop()
        game.last_move = game.moves[-1] if game.moves else None
        game.is_over = game.board.is_game_over()
        game.result = game.board.result() if game.is_over else None
        
        emit('move_undone', game.get_state(), broadcast=True)

@socketio.on('resign')
def on_resign(data):
    game_id = data.get('game_id')
    player = data.get('player')
    
    game_data = get_game(game_id)
    if not game_data:
        emit('error', {'message': 'Game not found'})
        return
    
    game = game_data['game']
    game.is_over = True
    game.result = '0-1' if player == 'white' else '1-0'
    
    emit('game_over', {
        'result': game.result,
        'reason': f'{player.capitalize()} resigned',
        **game.get_state()
    }, broadcast=True)

@socketio.on('request_analysis')
def on_request_analysis(data):
    game_id = data.get('game_id')
    emit('analysis_result', {
        'message': 'Analysis feature will be available after AlphaZero engine is trained',
        'game_id': game_id
    })

def check_game_ending(game):
    board = game.board
    
    if not board.is_game_over():
        if board.is_check():
            return {
                'in_check': True,
                'game_ending_reason': None
            }
        return {'in_check': False, 'game_ending_reason': None}
    
    if board.is_checkmate():
        winner = "Black" if board.turn else "White"
        return {
            'in_check': False,
            'game_ending_reason': 'checkmate',
            'winner': winner,
            'message': f'Checkmate! {winner} wins!'
        }
    elif board.is_stalemate():
        return {
            'in_check': False,
            'game_ending_reason': 'stalemate',
            'winner': None,
            'message': 'Stalemate! Game is a draw.'
        }
    elif board.is_insufficient_material():
        return {
            'in_check': False,
            'game_ending_reason': 'insufficient_material',
            'winner': None,
            'message': 'Draw by insufficient material.'
        }
    elif board.is_seventyfive_moves():
        return {
            'in_check': False,
            'game_ending_reason': 'seventy_five_moves',
            'winner': None,
            'message': 'Draw by 75-move rule.'
        }
    elif board.is_fivefold_repetition():
        return {
            'in_check': False,
            'game_ending_reason': 'fivefold_repetition',
            'winner': None,
            'message': 'Draw by fivefold repetition.'
        }
    else:
        return {
            'in_check': False,
            'game_ending_reason': 'other',
            'winner': None,
            'message': 'Game over.'
        }

def ai_make_move(game_id):
    game_data = get_game(game_id)
    if not game_data:
        return
    
    game = game_data['game']
    move = game.random_move()
    
    if move:
        game_ending_info = check_game_ending(game)
        socketio.emit('ai_move', {
            'move': move,
            **game.get_state(),
            **game_ending_info
        })

def ai_make_move_async(game_id):
    time.sleep(0.5)
    
    game_data = get_game(game_id)
    if not game_data:
        return
    
    game = game_data['game']
    move = game.random_move()
    
    if move:
        game_ending_info = check_game_ending(game)
        socketio.emit('ai_move', {
            'move': move,
            'evaluation': None,
            'simulations': None,
            **game.get_state(),
            **game_ending_info
        })

if __name__ == "__main__":
    socketio.run(app, host=app.config['HOST'], port=app.config['PORT'], debug=app.config['DEBUG'])
