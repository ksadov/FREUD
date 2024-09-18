from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import torch
import json
import os
from mlp_api import init_map, get_activation

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Global variable to store the activation_audio_map
global_activation_audio_map = None

"""
def init_map(layer_name: str, config: str, split: str) -> torch.Tensor:
    # TODO: Implement this function
    # For now, we'll return a dummy tensor
    return torch.randn(100, 100)
"""

"""
def get_activation(neuron_idx: int, audio_fname: str, activation_audio_map: dict) -> list:
    # TODO: Implement this function
    # For now, we'll return a dummy list of activations
    return [float(x) for x in torch.randn(100)]
"""


@app.route('/init', methods=['GET'])
def initialize():
    global global_activation_audio_map
    layer_name = "encoder.blocks.2.mlp.1"
    config = "/home/ksadov/whisper_sae/src/configs/tiny_mlp_2.json"
    with open(config, 'r') as f:
        config = json.load(f)
    split = "test-other"
    global_activation_audio_map = init_map(layer_name, config, split)
    return jsonify({"status": "Initialization complete"})


@app.route('/activation', methods=['GET'])
def activation():
    neuron_idx = int(request.args.get('neuron_idx', 0))
    audio_fname = "/home/ksadov/whisper_sae_dataset/LibriSpeech/test-other/8461/281231/8461-281231-0000.flac"
    activations = get_activation(
        neuron_idx, audio_fname, global_activation_audio_map)
    return jsonify({"activations": activations})


@app.route('/audio/<path:filename>', methods=['GET'])
def serve_audio(filename):
    audio_directory = "/home/ksadov/whisper_sae/activations/tiny_mlp_activations/test-other/encoder.blocks.2.mlp.1_neuron_20/"
    return send_file(os.path.join(audio_directory, filename), mimetype="audio/flac")


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
