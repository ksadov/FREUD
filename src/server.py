from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import torch
import json
import os
from src.feature_api import init_map, get_activation

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Global variable to store the activation_audio_map
global_activation_audio_map = None

DUMMY_AUDIO = "/home/ksadov/whisper_sae_dataset/LibriSpeech/test-other/8461/281231/8461-281231-0000.flac"


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
    activations = get_activation(
        neuron_idx, DUMMY_AUDIO, global_activation_audio_map)
    return jsonify({"activations": activations})


@app.route('/audio/<path:filename>', methods=['GET'])
def serve_audio(filename):
    return send_file(DUMMY_AUDIO, mimetype="audio/flac")


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
