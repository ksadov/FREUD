from typing import Optional, Tuple, Callable
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
from src.utils.constants import SAMPLE_RATE

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


class GlobalState:
    top_fn: Optional[Callable] = None
    n_features: Optional[int] = None
    layer_name: Optional[str] = None
    whisper_cache: Optional[WhisperActivationCache] = None
    sae_model: Optional[L1AutoEncoder | TopKAutoEncoder] = None
    whisper_subbed: Optional[WhisperSubbedActivation] = None


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
        whisper_cache = init_cache(
            config['whisper_model'], config['layer_name'], config['device'])
        sae_model = init_sae_from_checkpoint(
            config['sae_model']) if config['sae_model'] is not None else None
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
    whisper_subbed = init_subbed(
        config['whisper_model'], config['layer_name'], config['device'])
    activation_shape = dataloader.activation_shape
    n_features = activation_shape[-1]
    layer_name = config['layer_name']
    return (lambda feature_idx, n_files, max_val, min_val, absolute_magnitude, return_max_per_file:
            top_activations(dataloader, feature_idx, n_files, max_val,
                            min_val, absolute_magnitude, return_max_per_file),
            n_features, layer_name, whisper_cache, sae_model, whisper_subbed)


def get_top_activations(top_fn: Callable, feature_idx: int, n_files: int, max_val: Optional[float], min_val: Optional[float], absolute_magnitude: bool, return_max_per_file: bool) -> Tuple[list[str], list[torch.Tensor], list[float]]:
    top, max_per_file = top_fn(
        feature_idx, n_files, max_val, min_val, absolute_magnitude, return_max_per_file)
    top_files = [x[0] for x in top]
    activations = [x[1] for x in top]
    print("Got top activations.")
    return top_files, activations, max_per_file


def init_gui_data(config_path: str, from_disk: bool, files_to_search: Optional[int]):
    with open(config_path, 'r') as f:
        config = json.load(f)
    (GlobalState.top_fn, GlobalState.n_features, GlobalState.layer_name,
     GlobalState.whisper_cache, GlobalState.sae_model, GlobalState.whisper_subbed) = get_gui_data(config, from_disk, files_to_search)
    print("GUI data initialized.")


@app.route('/status', methods=['GET'])
def status():
    if GlobalState.top_fn is not None:
        return jsonify({"status": "Initialization complete", "n_features": GlobalState.n_features, "layer_name": GlobalState.layer_name})
    else:
        return jsonify({"status": "Initialization failed"}), 500


@app.route('/top_files', methods=['GET'])
def get_top_files():
    args = {
        'feature_idx': int(request.args.get('feature_idx', 0)),
        'n_files': int(request.args.get('n_files', 10)),
        'max_val': float(request.args.get('max_val')) if request.args.get('max_val') else None,
        'min_val': float(request.args.get('min_val')) if request.args.get('min_val') else None,
        'absolute_magnitude': request.args.get('absolute_magnitude', False),
        'return_max_per_file': True
    }
    top_files, activations, max_per_file = get_top_activations(
        GlobalState.top_fn, **args)
    return jsonify({"top_files": top_files, "activations": [x.tolist() for x in activations], "max_per_file": max_per_file})


@app.route('/audio/<path:filename>', methods=['GET'])
def serve_audio(filename):
    return send_file(f'/{filename}', mimetype="audio/flac")


def process_uploaded_audio(request):
    if 'audio' not in request.files:
        raise ValueError("No audio file provided")

    audio_file = request.files['audio']
    if audio_file.filename == '':
        raise ValueError("No selected file")

    audio, sr = sf.read(io.BytesIO(audio_file.read()))
    if sr != SAMPLE_RATE:
        # resample to 16 kHz
        resampled_len = int(len(audio) * SAMPLE_RATE / sr)
        audio = np.interp(np.linspace(0, len(audio) - 1,
                          resampled_len), np.arange(len(audio)), audio)

    return np.array(audio)


@app.route('/top_features', methods=['POST'])
def upload_and_get_top_features():
    try:
        audio_np = process_uploaded_audio(request)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    top_n = int(request.args.get('top_n', 32))

    top_indices, top_activations = top_activations_for_audio(
        audio_np, GlobalState.whisper_cache, GlobalState.sae_model, top_n)
    return jsonify({"top_indices": top_indices, "top_activations": [x.tolist() for x in top_activations]})


@app.route('/manipulate_feature', methods=['POST'])
def upload_and_manipulate_audio():
    try:
        audio_np = process_uploaded_audio(request)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    feat_idx = int(request.args.get('feat_idx', 0))
    manipulation_factor = float(request.args.get('manipulation_factor', 1.5))

    baseline_text, manipulated_text, standard_text, standard_activations, manipulated_activations = manipulate_latent(
        audio_np, GlobalState.whisper_cache, GlobalState.sae_model, GlobalState.whisper_subbed, feat_idx, manipulation_factor)

    return jsonify({
        "baseline_text": baseline_text,
        "manipulated_text": manipulated_text,
        "standard_text": standard_text,
        "standard_activations": standard_activations.tolist(),
        "manipulated_activations": manipulated_activations.tolist()
    })


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
