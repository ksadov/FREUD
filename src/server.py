import argparse
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import json
from feature_api import init_map, get_top_activations, get_batch_folder

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Global variable to store the activation_audio_map
global_activation_audio_map = None
global_batch_dir = None
global_init_at_start = False


def load_activation_map(config_path, layer_name, split, init_at_start):
    global global_activation_audio_map
    global global_batch_dir
    global global_init_at_start
    with open(config_path, 'r') as f:
        config = json.load(f)
    if init_at_start:
        global_init_at_start = True
        global_activation_audio_map = init_map(layer_name, config, split)
    else:
        global_batch_dir = get_batch_folder(config, split, layer_name)
    print("Activation map loaded successfully.")


@app.route('/status', methods=['GET'])
def status():
    if global_activation_audio_map is not None or not global_init_at_start:
        return jsonify({"status": "Initialization complete"})
    else:
        return jsonify({"status": "Initialization failed"}), 500


@app.route('/top_files', methods=['GET'])
def get_top_files():
    neuron_idx = int(request.args.get('neuron_idx', 0))
    n_files = int(request.args.get('n_files', 10))
    top_files, activations = get_top_activations(
        global_activation_audio_map, global_batch_dir, n_files, neuron_idx)
    activations = [x.tolist() for x in activations]
    return jsonify({"top_files": top_files, "activations": activations})


@app.route('/audio/<path:filename>', methods=['GET'])
def serve_audio(filename):
    # filename is global path to audio file
    global_fname = '/' + filename
    return send_file(global_fname, mimetype="audio/flac")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--layer_name', type=str, required=True)
    parser.add_argument('--split', type=str, required=True)
    parser.add_argument('--init_at_start', action='store_true')
    args = parser.parse_args()
    load_activation_map(args.config, args.layer_name, args.split, args.init_at_start)
    app.run(debug=True, host='0.0.0.0', port=5555)
