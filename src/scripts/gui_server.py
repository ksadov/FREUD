from typing import Optional
import torch
import argparse
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import json

from src.utils.activations import make_top_fn

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Global variable to store the activation_audio_map
top_fn = None

def get_top_activations(top_fn: callable,
                        neuron_idx: int,
                        n_files: int,
                        max_val: Optional[float] = None
                        ) -> tuple[list[str], list[torch.Tensor]]:
    top = top_fn(neuron_idx, n_files, max_val)
    top_files = [x[0] for x in top]
    activations = [x[1] for x in top]
    print("Got top activations.")
    return top_files, activations

def load_activation_map(config_path, from_disk, files_to_search):
    global top_fn
    with open(config_path, 'r') as f:
        config = json.load(f)
    top_fn = make_top_fn(config, from_disk, files_to_search)
    print("Function to find top activations loaded successfully.")


@app.route('/status', methods=['GET'])
def status():
    if top_fn is not None:
        return jsonify({"status": "Initialization complete"})
    else:
        return jsonify({"status": "Initialization failed"}), 500


@app.route('/top_files', methods=['GET'])
def get_top_files():
    neuron_idx = int(request.args.get('neuron_idx', 0))
    n_files = int(request.args.get('n_files', 10))
    max_val_arg = request.args.get('max_val', None)
    max_val = float(max_val_arg) if max_val_arg is not None else None
    top_files, activations = get_top_activations(top_fn, neuron_idx, n_files, max_val)
    activations = [x.tolist() for x in activations]
    return jsonify({"top_files": top_files, "activations": activations})


@app.route('/audio/<path:filename>', methods=['GET'])
def serve_audio(filename):
    # filename is global path to audio file
    global_fname = '/' + filename
    return send_file(global_fname, mimetype="audio/flac")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to feature configuration file')
    parser.add_argument('--from_disk', action='store_true', help='Whether to load activations from disk')
    parser.add_argument('--files_to_search', type=int, default=None, 
                        help='Number of files to search (None to search all)')
    args = parser.parse_args()
    load_activation_map(args.config, args.from_disk, args.files_to_search)
    app.run(debug=True, host='0.0.0.0', port=5555)
