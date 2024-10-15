from typing import Optional
import torch
import argparse
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import json
import soundfile as sf
import io
import numpy as np

from src.dataset.activations import MemoryMappedActivationDataLoader, FlyActivationDataLoader, init_sae_from_checkpoint
from src.utils.activations import top_activations, top_activations_for_audio, manipulate_latent
from src.models.hooked_model import init_cache, WhisperActivationCache, init_subbed, WhisperSubbedActivation
from src.models.l1autoencoder import L1AutoEncoder
from src.models.topkautoencoder import TopKAutoEncoder

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Global variable to store the activation_audio_map
top_fn = None
n_features = None
layer_name = None
whisper_cache = None
sae_model = None
whisper_subbed = None


def get_gui_data(config: dict, from_disk: bool, files_to_search: Optional[int]) -> \
    tuple[callable, int, str, WhisperActivationCache, L1AutoEncoder | TopKAutoEncoder, WhisperSubbedActivation]:
    if from_disk:
        dataloader = MemoryMappedActivationDataLoader(
            config['out_folder'],
            config['layer_name'],
            config['batch_size'],
            dl_max_workers=config['dl_max_workers'],
            subset_size=files_to_search
        )
        whisper_cache = init_cache(config['whisper_model'], config['layer_name'], config['device'])
        sae_model = init_sae_from_checkpoint(config['sae_model'])
    else:
        dataloader = FlyActivationDataLoader(
            config['data_path'],
            config['whisper_model'],
            config['sae_model'],
            config['layer_name'],
            config['device'],
            config['batch_size'],
            dl_max_workers=config['dl_max_workers'],
            subset_size=files_to_search
        )
        whisper_cache = dataloader.whisper_cache
        sae_model = dataloader.sae_model
    whisper_subbed = init_subbed(config['whisper_model'], config['layer_name'], config['device'])
    activation_shape = dataloader.activation_shape
    n_features = activation_shape[-1]
    layer_name = config['layer_name']
    return (lambda neuron_idx, n_files, max_val, min_val, absolute_magnitude, return_max_per_file:
            top_activations(dataloader, neuron_idx, n_files, max_val,
                            min_val, absolute_magnitude, return_max_per_file),
            n_features, layer_name, whisper_cache, sae_model, whisper_subbed)


def get_top_activations(top_fn: callable,
                        neuron_idx: int,
                        n_files: int,
                        max_val: Optional[float],
                        min_val: Optional[float],
                        absolute_magnitude: bool,
                        return_max_per_file: bool
                        ) -> tuple[list[str], list[torch.Tensor]]:
    top, max_per_file = top_fn(
        neuron_idx, n_files, max_val, min_val, absolute_magnitude, return_max_per_file)
    top_files = [x[0] for x in top]
    activations = [x[1] for x in top]
    print("Got top activations.")
    return top_files, activations, max_per_file


def init_gui_data(config_path, from_disk, files_to_search):
    global top_fn
    global n_features
    global layer_name
    global whisper_cache
    global sae_model
    global whisper_subbed
    with open(config_path, 'r') as f:
        config = json.load(f)
    top_fn, n_features, layer_name, whisper_cache, sae_model, whisper_subbed = get_gui_data(
        config, from_disk, files_to_search)
    print("GUI data initialized.")


@app.route('/status', methods=['GET'])
def status():
    if top_fn is not None:
        return jsonify({"status": "Initialization complete", "n_features": n_features, "layer_name": layer_name})
    else:
        return jsonify({"status": "Initialization failed"}), 500


@app.route('/top_files', methods=['GET'])
def get_top_files():
    neuron_idx = int(request.args.get('neuron_idx', 0))
    n_files = int(request.args.get('n_files', 10))
    max_val_arg = request.args.get('max_val', None)
    min_val_arg = request.args.get('min_val', None)
    absolute_magnitude = request.args.get('absolute_magnitude', False)
    max_val = float(max_val_arg) if max_val_arg is not None else None
    min_val = float(min_val_arg) if min_val_arg is not None else None
    return_max_per_file = True
    top_files, activations, max_per_file = get_top_activations(top_fn, neuron_idx, n_files, max_val, min_val,
                                                               absolute_magnitude, return_max_per_file)
    activations = [x.tolist() for x in activations]
    return jsonify({"top_files": top_files, "activations": activations, "max_per_file": max_per_file})


@app.route('/audio/<path:filename>', methods=['GET'])
def serve_audio(filename):
    # filename is global path to audio file
    global_fname = '/' + filename
    return send_file(global_fname, mimetype="audio/flac")

@app.route('/analyze_audio', methods=['POST'])
def upload_and_analyze_audio():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    
    top_n = request.args.get('top_n', 32)
    top_n = int(top_n)
    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if audio_file:
        # Read the audio file
        audio_data, sample_rate = sf.read(io.BytesIO(audio_file.read()))

        # Convert to numpy array if it's not already
        audio_np = np.array(audio_data)

        top_indices, top_activations = top_activations_for_audio(audio_np, whisper_cache, sae_model, top_n)
        top_activations = [x.tolist() for x in top_activations]

        # Return the result
        return jsonify({"top_indices": top_indices, "top_activations": top_activations})

    return jsonify({"error": "Failed to process audio file"}), 500

@app.route('/manipulate_feature', methods=['POST'])
def upload_and_manipulate_audio():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    
    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if audio_file:
        # Read the audio file
        audio_data, sample_rate = sf.read(io.BytesIO(audio_file.read()))

        # Convert to numpy array if it's not already
        audio_np = np.array(audio_data)

        feat_idx = request.args.get('feat_idx', 0)
        feat_idx = int(feat_idx)
        manipulation_factor = request.args.get('manipulation_factor', 1.5)
        manipulation_factor = float(manipulation_factor)

        baseline_text, manipulated_text, standard_text, standard_activations, manipulated_activations = manipulate_latent(
            audio_np, whisper_cache, sae_model, whisper_subbed, feat_idx, manipulation_factor)

        print("manipulated activations: ", manipulated_activations)
        print("Manipulate activation shape: ", manipulated_activations.shape)
        print("standard activations: ", standard_activations)
        # Return the result
        return jsonify({"baseline_text": baseline_text, "manipulated_text": manipulated_text, 
                        "standard_text": standard_text, "standard_activations": standard_activations.tolist(),
                        "manipulated_activations": manipulated_activations.tolist()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='Path to feature configuration file')
    parser.add_argument('--from_disk', action='store_true',
                        help='Whether to load activations from disk')
    parser.add_argument('--files_to_search', type=int, default=None,
                        help='Number of files to search (None to search all)')
    args = parser.parse_args()
    init_gui_data(args.config, args.from_disk, args.files_to_search)
    app.run(debug=True, host='0.0.0.0', port=5555)
