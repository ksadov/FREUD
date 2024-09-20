from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import json
from feature_api import init_map, get_top_activations

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Global variable to store the activation_audio_map
global_activation_audio_map = None


def load_activation_map():
    global global_activation_audio_map
    layer_name = "encoder.blocks.2.mlp.1"
    config_path = "/home/ksadov/whisper_sae/src/configs/tiny_mlp_2.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    split = "test-other"
    global_activation_audio_map = init_map(layer_name, config, split)
    print("Activation map loaded successfully.")


@app.route('/status', methods=['GET'])
def status():
    if global_activation_audio_map is not None:
        return jsonify({"status": "Initialization complete"})
    else:
        return jsonify({"status": "Initialization failed"}), 500


@app.route('/top_files', methods=['GET'])
def get_top_files():
    print("getting top files")
    neuron_idx = int(request.args.get('neuron_idx', 0))
    print("Neuron idx", neuron_idx)
    n_files = int(request.args.get('n_files', 10))
    print("N files", n_files)
    top_files, activations = get_top_activations(
        global_activation_audio_map, n_files, neuron_idx)
    activations = [x.tolist() for x in activations]
    # top_files = []
    return jsonify({"top_files": top_files, "activations": activations})


@app.route('/audio/<path:filename>', methods=['GET'])
def serve_audio(filename):
    # filename is global path to audio file
    global_fname = '/' + filename
    return send_file(global_fname, mimetype="audio/flac")


if __name__ == '__main__':
    load_activation_map()
    app.run(debug=True, host='0.0.0.0')
