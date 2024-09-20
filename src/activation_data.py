import torch
import os
from librispeech_data import LibriSpeechDataset
import json
import argparse
import matplotlib.pyplot as plt
import torchaudio
from feature_api import load_activations, id_audio_assoc, activation_audio_assoc, top_activating_files
from constants import TIMESTEP_S


def activation_statistics(activation_audio_map: dict) -> dict:
    activation_shape = list(activation_audio_map.keys())[0].shape
    n_neurons = activation_shape[-1]
    # mean per neuron
    neuron_means = torch.zeros(n_neurons)
    neuron_stds = torch.zeros(n_neurons)
    for activation in activation_audio_map.keys():
        neuron_means += activation.mean(dim=0)
        neuron_stds += activation.std(dim=0)
    neuron_means /= len(activation_audio_map)
    neuron_stds /= len(activation_audio_map)
    return {
        "mean": neuron_means,
        "std": neuron_stds,
    }


def plot_statistics(stats: dict, plot_prefix: str):
    # plot mean and std separately
    plt.figure()
    plt.plot(stats["mean"])
    plt.title("Mean activation per neuron")
    mean_plot_filename = f"{plot_prefix}_mean.png"
    plt.savefig(mean_plot_filename)
    print("Plot saved to", mean_plot_filename)
    plt.figure()
    plt.plot(stats["std"])
    plt.title("Std activation per neuron")
    std_plot_filename = f"{plot_prefix}_std.png"
    plt.savefig(std_plot_filename)
    print("Plot saved to", std_plot_filename)


def plot_audio_waveform(activation, neuron_idx: int, plot_prefix: str, audio_file: str):
    per_neuron_activation = activation.transpose(0, 1)[neuron_idx]
    activation = per_neuron_activation
    waveform, _ = torchaudio.load(audio_file)
    waveform = waveform[0]
    waveform_time = torch.arange(0, waveform.shape[0]) / 16000
    # trim activation to match waveform length
    activation_len = int(waveform.shape[0] // (16000 * TIMESTEP_S))
    activation_time = torch.arange(0, activation_len * TIMESTEP_S, TIMESTEP_S)
    activation = activation[:activation_len]
    # create image of two plots: activation and audio waveform
    fig, ax1 = plt.subplots()
    ax1.plot(waveform_time, waveform, 'b-')
    ax1.set_xlabel('time (s)')
    ax1.set_ylabel('audio waveform', color='b')
    ax2 = ax1.twinx()
    ax2.plot(activation_time, activation, 'r-')
    ax2.set_ylabel('activation', color='r')
    plt.title("Audio waveform and activation")
    plot_filename = f"{plot_prefix}_neuron_{neuron_idx}_{os.path.basename(audio_file)}.png"
    plt.savefig(plot_filename)
    print("Plot saved to", plot_filename)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="tiny_mlp_2.json")
    parser.add_argument("--split", type=str, default="test-other")
    parser.add_argument("--layer", type=str, default="encoder.blocks.2.mlp.1")
    parser.add_argument("--neuron", type=int, default=20)
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = json.load(f)
    batch_folder = f"{config['out_folder_prefix']}/{args.split}/{args.layer}"
    activation_map = load_activations(batch_folder)
    dataset = LibriSpeechDataset(
        config["data_path"], args.split, torch.device(config["device"]))
    id_audio_map = id_audio_assoc(dataset)
    activation_audio_map = activation_audio_assoc(activation_map, id_audio_map)
    stats = activation_statistics(activation_audio_map)
    plot_prefix = f"{config['out_folder_prefix']}/{args.split}/{args.layer}"
    plot_statistics(stats, plot_prefix)
    top_files = top_activating_files(activation_audio_map, 10, args.neuron)
    print("Top files", top_files)
    # copy top files to a folder
    top_files_folder = f"{config['out_folder_prefix']}/{args.split}/{args.layer}_neuron_{args.neuron}"
    os.makedirs(top_files_folder, exist_ok=True)
    for _, audio_file, _, _ in top_files:
        os.system(f"cp {audio_file} {top_files_folder}")
    print("Top files copied to", top_files_folder)
    # plot audio waveform
    audio_file = top_files[0]
    plot_audio_waveform(audio_file[-1], args.neuron,
                        plot_prefix, audio_file[1])


if __name__ == "__main__":
    main()
