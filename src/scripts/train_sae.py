
from typing import Optional
import torch
import whisper
from torch.amp import autocast
from tqdm import tqdm
from src.dataset.activations import FlyActivationDataLoader, MemoryMappedActivationDataLoader
from src.utils.audio_utils import get_mels_from_audio_path
from src.models.hooked_model import WhisperSubbedActivation
import numpy as np
import random
import os
from src.models.l1autoencoder import L1AutoEncoder
from pathlib import Path
from torch.optim import RAdam
from torch.optim.lr_scheduler import CosineAnnealingLR
import gc
from functools import partial
from time import perf_counter
import argparse
import json
import torchaudio
from contextlib import nullcontext
import time

N_TRANSCRIPTS = 4


def init_dataloader(from_disk: bool, data_path: str, whisper_model: str, sae_checkpoint: str, layer_name: str,
                    device: torch.device, batch_size: int, dl_max_workers: int, subset_size: Optional[int]):
    if from_disk:
        loader = MemoryMappedActivationDataLoader(
            data_path=data_path,
            layer_name=layer_name,
            batch_size=batch_size,
            dl_max_workers=dl_max_workers,
            subset_size=subset_size
        )
    else:
        loader = FlyActivationDataLoader(
            data_path=data_path,
            whisper_model=whisper_model,
            sae_checkpoint=sae_checkpoint,
            layer_name=layer_name,
            device=device,
            batch_size=batch_size,
            dl_max_workers=dl_max_workers,
            subset_size=subset_size
        )
    feat_dim = loader.activation_shape[-1]
    dset_len = loader.dataset_length
    return loader, feat_dim, dset_len


def validate(
    model: L1AutoEncoder,
    val_folder: str,
    device: torch.device,
    layer_name: str,
    whisper_model_name: str,
    log_base_transcripts: bool,
    from_disk: bool
):
    model.eval()
    whisper_model = whisper.load_model(whisper_model_name)
    whisper_sub = WhisperSubbedActivation(
        model=whisper_model,
        substitution_layer=layer_name,
        device=device
    )
    losses_recon = []
    losses_l1 = []
    subbed_transcripts = []
    base_transcripts = []
    base_filenames = []

    val_loader, _, _ = init_dataloader(
        from_disk, val_folder, whisper_model_name, None, layer_name, device, 1, 1, None)
    encoded_magnitude_values = torch.zeros(
        (len(val_loader), model.n_dict_components)).to(device)
    context_manager = autocast(device_type=str(
        device)) if device == torch.device("cuda") else nullcontext()

    for i, datapoints in tqdm(enumerate(val_loader), total=len(val_loader)):
        with torch.no_grad(), context_manager:
            activations, filenames = datapoints
            activations = activations.to(device)
            filenames = filenames[0]
            out = model(activations)
            detached_encoded = torch.abs(out.encoded.detach()).squeeze()
            encoded_max = torch.max(detached_encoded, dim=0).values
            encoded_magnitude_values[i] = encoded_max
            losses_recon.append(out.reconstruction_loss.item())
            losses_l1.append(out.l1_loss.item())
            if i < N_TRANSCRIPTS:
                mels = get_mels_from_audio_path(device, filenames)
                subbed_result = whisper_sub.forward(mels, out.sae_out)
                subbed_transcripts.append(subbed_result.text)
                if log_base_transcripts:
                    base_result = whisper_sub.forward(mels, None)
                    base_transcripts.append(base_result.text)
                    base_filenames.append(filenames)

    model.train()
    print("Calculating means...")
    encoded_mag_means = torch.mean(
        encoded_magnitude_values, dim=0).cpu().numpy()
    print("Calculating stds...")
    encoded_mag_stds = torch.std(encoded_magnitude_values, dim=0).cpu().numpy()
    return (np.array(losses_recon).mean(), np.array(losses_l1).mean(), subbed_transcripts, base_transcripts,
            base_filenames, encoded_mag_means, encoded_mag_stds)


def set_seeds(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def save_checkpoint(state, save_path):
    """
    Consumes a generic state dictionary. Unpacks state_dict
    for each element of state if required.
    """

    if "model" in state:
        # we need to call state_dict() on all ranks in case it is calling all_gather
        model = state["model"]

    checkpoint = {}
    for k, v in state.items():
        if hasattr(v, "state_dict"):
            checkpoint[k] = v.state_dict()
        else:
            checkpoint[k] = v
    torch.save(checkpoint, save_path)

    if "model" in state:
        state["model"] = model


def prepare_tb_logging(path=None):
    """
    Ensures that the dir for logging exists and returns a tensorboard logger.
    """
    from torch.utils.tensorboard import SummaryWriter  # dot

    logdir_path = Path(path)
    logdir_path.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(logdir_path, flush_secs=10)


def load_checkpoint(
    state,
    load_path,
    device,
):
    """
    Updates a generic state dictionary. Takes the items in 'checkpoint', and pushes them
    into the preloaded state values
    """
    checkpoint = torch.load(load_path, map_location=device)
    for k, v in state.items():
        if hasattr(v, "load_state_dict"):
            v.load_state_dict(checkpoint[k])
        else:
            state[k] = checkpoint[k]
    del checkpoint
    if "numpy_rng_state" in state:
        np.random.set_state(state["numpy_rng_state"])
    if "torch_rng_state" in state:
        torch.set_rng_state(state["torch_rng_state"])
    if "random_rng_state" in state:
        random.setstate(state["random_rng_state"])
    if "cuda_rng_state" in state:
        torch.cuda.set_rng_state(state["cuda_rng_state"])
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.is_autocast_enabled():
        torch.clear_autocast_cache()

    gc.collect()


def train(seed: int,
          train_folder: str,
          val_folder: str,
          device: torch.device,
          n_dict_components: int,
          run_dir: str,
          lr: float,
          weight_decay: float,
          steps: int,
          clip_thresh: float,
          batch_size: int,
          dl_max_workers: int,
          log_tb_every: int,
          save_every: int,
          val_every: int,
          checkpoint: str,
          recon_alpha: float,
          layer_name: str,
          whisper_model: str,
          from_disk: bool,
          ):
    set_seeds(seed)
    train_loader, feat_dim, dset_len = init_dataloader(
        from_disk, train_folder, whisper_model, None, layer_name, device, batch_size, dl_max_workers, None)

    hparam_dict = {
        "lr": lr,
        "weight_decay": weight_decay,
        "steps": steps,
        "clip_thresh": clip_thresh,
        "batch_size": batch_size,
        "recon_alpha": recon_alpha,
        "n_dict_components": n_dict_components,
        "layer_name": layer_name,
        "whisper_model": whisper_model,
        "activation_size": feat_dim,
        "train_folder": train_folder,
        "val_folder": val_folder,
    }

    model = L1AutoEncoder(hparam_dict).to(device)
    dist_model = model

    os.makedirs(run_dir, exist_ok=True)
    checkpoint_out_dir = run_dir + "/checkpoints"
    os.makedirs(checkpoint_out_dir, exist_ok=True)

    tb_logger = prepare_tb_logging(run_dir)
    tb_logger.add_text("hparams", json.dumps(hparam_dict, indent=4))
    model_out = run_dir + "/model"
    print("Model: %.2fM" % (sum(p.numel()
          for p in model.parameters()) / 1.0e6))
    logged_base_transcripts = False

    optimizer = RAdam(
        dist_model.parameters(), eps=1e-5, lr=lr, weight_decay=weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=steps, eta_min=0)

    state = {
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "step": 0,
        "best_val_loss": float("inf"),
        "hparams": hparam_dict,
    }
    meta = {}
    meta["effective_batch_size"] = batch_size
    meta["model_params"] = sum(x.numel() for x in dist_model.parameters())

    if checkpoint:
        print(f"Checkpoint: {checkpoint}")
        load_checkpoint(state, checkpoint, device=device)

    while state["step"] < steps:
        model.train()
        pbar = tqdm(enumerate(train_loader), total=len(
            train_loader), desc=f"Training")

        for batch_idx, (activations, _) in pbar:
            activations = activations.to(device)

            optimizer.zero_grad()

            with autocast(str(device)):
                out = dist_model(activations)
                loss = out.reconstruction_loss + out.l1_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                dist_model.parameters(), clip_thresh)
            optimizer.step()
            scheduler.step()

            state["step"] += 1

            pbar.set_postfix({
                'loss': f"{loss.item():.3f}",
                'step': state["step"]
            })

            meta["loss_recon"] = out.reconstruction_loss.item()
            meta["loss_l1"] = out.l1_loss.item()

            if state["step"] % log_tb_every == 0:
                tb_logger.add_scalar("train/loss", loss, state["step"])
                tb_logger.add_scalar("train/loss_recon",
                                     meta["loss_recon"], state["step"])
                tb_logger.add_scalar(
                    "train/loss_l1", meta["loss_l1"], state["step"])
                tb_logger.add_scalar(
                    "train/lr", scheduler.get_last_lr()[0], state["step"])

            if state["step"] % save_every == 0:
                save_checkpoint(state, checkpoint_out_dir +
                                "/step" + str(state["step"]) + ".pth")

            if state["step"] % val_every == 0:
                print("Validating...")
                val_loss_recon, val_loss_l1, subbed_transcripts, base_transcripts, base_filenames, \
                    encoded_mag_means, encoded_mag_stds = validate(
                        model, val_folder, device, layer_name,
                        whisper_model, not logged_base_transcripts, from_disk
                    )
                logged_base_transcripts = True
                print(
                    f"{state['step']} validation, loss_recon={val_loss_recon:.3f}")

                tb_logger.add_scalar(
                    "val/loss_recon", val_loss_recon, state["step"])
                tb_logger.add_scalar("val/loss_l1", val_loss_l1, state["step"])

                tb_logger.add_histogram(
                    "val/encoded/magnitude_means", np.array(encoded_mag_means), state["step"])
                tb_logger.add_histogram(
                    "val/encoded/magnitude_stds", np.array(encoded_mag_stds), state["step"])

                for i, transcript in enumerate(subbed_transcripts):
                    tb_logger.add_text(
                        f"val/transcripts/reconstructed_{i}", transcript, state["step"])

                if base_transcripts:
                    for i, transcript in enumerate(base_transcripts):
                        tb_logger.add_text(
                            f"val/transcripts/base_{i}", transcript, state["step"])
                    for i, filename in enumerate(base_filenames):
                        audio = torchaudio.load(filename)[0]
                        tb_logger.add_audio(
                            f"val/transcripts/audio_{i}", audio, state["step"], sample_rate=16000)

                if val_loss_recon < state["best_val_loss"]:
                    print("Saving new best validation")
                    state["best_val_loss"] = val_loss_recon
                    save_checkpoint(state, checkpoint_out_dir + "/bestval.pth")
                    pytorch_model_path = model_out[:-3] + ".bestval"
                    torch.save(model, pytorch_model_path)

            if state["step"] >= steps:
                break

        save_checkpoint(state, checkpoint_out_dir +
                        "/step" + str(state["step"]) + ".pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        help="Path to train configuration file")
    args = parser.parse_args()
    # load config json
    with open(args.config, "r") as f:
        config = json.load(f)
    config["device"] = torch.device(config["device"])
    train(**config)
