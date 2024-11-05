from typing import Optional
import torch
import whisper
from torch.amp import autocast
from tqdm import tqdm
from src.dataset.activations import (
    FlyActivationDataLoader,
    MemoryMappedActivationDataLoader,
)
from src.utils.audio_utils import get_mels_from_audio_path
from src.utils.constants import get_n_mels
from src.models.hooked_model import WhisperSubbedActivation
import numpy as np
import random
import os
from src.models.config import L1AutoEncoderConfig, TopKAutoEncoderConfig
from src.models.l1autoencoder import L1AutoEncoder, L1ForwardOutput
from src.models.topkautoencoder import TopKAutoEncoder, TopKForwardOutput
from pathlib import Path
from torch.optim import RAdam, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import get_linear_schedule_with_warmup
import gc
import argparse
import json
import torchaudio
from contextlib import nullcontext

N_TRANSCRIPTS = 4


def init_dataloader(
    from_disk: bool,
    data_path: str,
    whisper_model: str,
    sae_checkpoint: str,
    layer_name: str,
    device: torch.device,
    batch_size: int,
    dl_max_workers: int,
    subset_size: Optional[int],
    dl_kwargs: dict,
):
    if from_disk:
        loader = MemoryMappedActivationDataLoader(
            data_path=data_path,
            layer_name=layer_name,
            batch_size=batch_size,
            dl_max_workers=dl_max_workers,
            subset_size=subset_size,
            dl_kwargs=dl_kwargs,
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
            subset_size=subset_size,
            dl_kwargs=dl_kwargs,
        )
    feat_dim = loader.activation_shape[-1]
    dset_len = loader.dataset_length
    return loader, feat_dim, dset_len


def topk_feature_extraction(out, mag_vals_dim, batch_idx=None, device=None):
    # Get the encoded activations and indices
    detached_encoded = torch.abs(out.encoded.top_acts.detach()).squeeze()
    detached_idxes = out.encoded.top_indices.squeeze()

    # Create result tensor
    if batch_idx is not None:
        encoded_magnitude_values = torch.zeros(mag_vals_dim).to(device)
    else:
        encoded_magnitude_values = torch.zeros(
            (detached_encoded.shape[0], mag_vals_dim)
        ).to(device)

    # Create a mask for each feature index
    time_steps = detached_encoded.shape[0]

    # Vectorized approach using boolean masking
    feature_range = torch.arange(mag_vals_dim).to(device)

    # Expand dimensions for broadcasting
    # [time_steps, num_indices, 1]
    detached_idxes_expanded = detached_idxes.unsqueeze(-1)
    feature_range_expanded = feature_range.view(1, 1, -1)  # [1, 1, mag_vals_dim]

    # Create masks for each feature
    # [time_steps, num_indices, mag_vals_dim]
    masks = detached_idxes_expanded == feature_range_expanded

    # Use the masks to gather relevant values
    # Expand detached_encoded for broadcasting
    detached_encoded_expanded = detached_encoded.unsqueeze(-1).expand(
        -1, -1, mag_vals_dim
    )

    # Apply masks and get maximum values
    masked_values = torch.where(
        masks, detached_encoded_expanded, torch.tensor(-float("inf"))
    )
    max_values = masked_values.max(dim=0)[0].max(dim=0)[0]

    # Replace -inf with 0 for features that weren't present
    max_values = torch.where(max_values == -float("inf"), torch.tensor(0.0), max_values)

    if batch_idx is not None:
        encoded_magnitude_values = max_values
    else:
        encoded_magnitude_values[batch_idx] = max_values

    return encoded_magnitude_values


def validate(
    model: L1AutoEncoder | TopKAutoEncoder,
    val_folder: str,
    device: torch.device,
    layer_name: str,
    whisper_model_name: str,
    log_base_transcripts: bool,
    from_disk: bool,
):
    model.eval()
    whisper_model = whisper.load_model(whisper_model_name)
    whisper_sub = WhisperSubbedActivation(
        model=whisper_model, substitution_layer=layer_name, device=device
    )
    losses_recon = []
    losses_l1 = []
    fvus = []
    losses_auxk = []
    multi_topk_fvu = []
    subbed_transcripts = []
    base_transcripts = []
    base_filenames = []
    mses = []

    dl_kwargs = {
        "shuffle": False,
    }
    val_loader, _, _ = init_dataloader(
        from_disk,
        val_folder,
        whisper_model_name,
        None,
        layer_name,
        device,
        1,
        1,
        None,
        dl_kwargs,
    )
    mag_vals_dim = model.n_dict_components
    encoded_magnitude_values = torch.zeros((len(val_loader), mag_vals_dim)).to(device)
    context_manager = (
        autocast(device_type=str(device))
        if device == torch.device("cuda")
        else nullcontext()
    )

    for i, datapoints in tqdm(enumerate(val_loader), total=len(val_loader)):
        with torch.no_grad(), context_manager:
            activations, filenames = datapoints
            activations = activations.to(device)
            filenames = filenames[0]
            out, mse = model(activations, return_mse=True)
            mses.append(mse.item())
            if isinstance(out, L1ForwardOutput):
                detached_encoded = torch.abs(out.encoded.latent.detach()).squeeze()
                encoded_max = torch.max(detached_encoded, dim=0).values
                encoded_magnitude_values[i] = encoded_max
            elif isinstance(out, TopKForwardOutput):
                encoded_magnitude_values[i] = topk_feature_extraction(
                    out, mag_vals_dim, 1, device
                )

            if isinstance(model, L1AutoEncoder):
                losses_recon.append(out.reconstruction_loss.item())
                losses_l1.append(out.l1_loss.item())
            elif isinstance(model, TopKAutoEncoder):
                fvus.append(out.fvu.item())
                losses_auxk.append(out.auxk_loss.item())
                multi_topk_fvu.append(out.multi_topk_fvu.item())
            if i < N_TRANSCRIPTS:
                n_mels = get_n_mels(whisper_model_name)
                mels = get_mels_from_audio_path(device, filenames, n_mels)
                subbed_result = whisper_sub.forward(mels, out.sae_out)
                subbed_transcripts.append(subbed_result.text)
                if log_base_transcripts:
                    base_result = whisper_sub.forward(mels, None)
                    base_transcripts.append(base_result.text)
                    base_filenames.append(filenames)

    model.train()
    print("Calculating means...")
    encoded_mag_maxes = torch.max(encoded_magnitude_values, dim=0)[0].cpu().numpy()
    print("Calculating stds...")
    encoded_mag_stds = torch.std(encoded_magnitude_values, dim=0).cpu().numpy()
    losses_dict = {
        "l1": np.array(losses_l1).mean() if losses_l1 else None,
        "recon": np.array(losses_recon).mean() if losses_recon else None,
        "fvu": np.array(fvus).mean() if fvus else None,
        "auxk_loss": np.array(losses_auxk).mean() if losses_auxk else None,
        "multi_topk_fvu": np.array(multi_topk_fvu).mean() if multi_topk_fvu else None,
        "mse": np.array(mses).mean(),
    }
    return (
        losses_dict,
        subbed_transcripts,
        base_transcripts,
        base_filenames,
        encoded_mag_maxes,
        encoded_mag_stds,
    )


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


def train(
    seed: int,
    train_folder: str,
    val_folder: str,
    device: torch.device,
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
    start_checkpoint: str,
    whisper_config: dict,
    optimizer: str,
    scheduler: str,
    scheduler_params: dict,
    from_disk: bool,
    autoencoder_variant: str,
    autoencoder_config: dict,
):
    set_seeds(seed)
    dl_kwargs = {"shuffle": True, "drop_last": True}
    train_loader, feat_dim, dset_len = init_dataloader(
        from_disk,
        train_folder,
        whisper_config["model"],
        None,
        whisper_config["layer_name"],
        device,
        batch_size,
        dl_max_workers,
        None,
        dl_kwargs,
    )

    hparam_dict = {
        "autoencoder_variant": autoencoder_variant,
        "autoencoder_config": autoencoder_config,
        "lr": lr,
        "weight_decay": weight_decay,
        "steps": steps,
        "clip_thresh": clip_thresh,
        "batch_size": batch_size,
        "whisper_config": whisper_config,
        "activation_size": feat_dim,
        "train_folder": train_folder,
        "val_folder": val_folder,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "scheduler_params": scheduler_params,
    }
    assert autoencoder_variant in [
        "l1",
        "topk",
    ], f"Invalid autoencoder variant: {autoencoder_variant}, must be 'l1' or 'topk'"
    if autoencoder_variant == "l1":
        cfg = L1AutoEncoderConfig.from_dict(autoencoder_config)
        model = L1AutoEncoder(activation_size=feat_dim, cfg=cfg)
    else:
        cfg = TopKAutoEncoderConfig.from_dict(autoencoder_config)
        model = TopKAutoEncoder(activation_size=feat_dim, cfg=cfg)
    dist_model = model.to(device)

    os.makedirs(run_dir, exist_ok=True)
    checkpoint_out_dir = run_dir + "/checkpoints"
    os.makedirs(checkpoint_out_dir, exist_ok=True)

    tb_logger = prepare_tb_logging(run_dir)
    tb_logger.add_text("hparams", json.dumps(hparam_dict, indent=4))
    model_out = run_dir + "/model"
    print("Model: %.2fM" % (sum(p.numel() for p in model.parameters()) / 1.0e6))
    logged_base_transcripts = False

    if optimizer == "radam":
        optimizer = RAdam(
            dist_model.parameters(), eps=1e-5, lr=lr, weight_decay=weight_decay
        )
    elif optimizer == "adam":
        optimizer = Adam(dist_model.parameters(), lr=lr)
    else:
        raise ValueError(f"Invalid optimizer: {optimizer}, must be 'radam' or 'adam'")

    if scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=steps, eta_min=0)
    elif scheduler == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=scheduler_params["num_warmup_steps"],
            num_training_steps=steps,
        )
    else:
        raise ValueError(
            f"Invalid scheduler: {scheduler}, must be 'cosine' or 'linear'"
        )

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

    if start_checkpoint is not None:
        print(f"Checkpoint: {start_checkpoint}")
        load_checkpoint(state, start_checkpoint, device=device)

    if isinstance(model, TopKAutoEncoder):
        num_frames_since_fired = torch.zeros(
            model.n_dict_components, device=device, dtype=torch.long
        )

    while state["step"] < steps:
        model.train()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Training")

        for batch_idx, (activations, _) in pbar:
            activations = activations.to(device)

            if isinstance(model, TopKAutoEncoder):
                did_fire = torch.zeros(
                    model.n_dict_components, device=device, dtype=torch.bool
                )

            optimizer.zero_grad()

            with autocast(str(device)):
                if autoencoder_variant == "l1":
                    out = dist_model(activations)
                    loss = out.reconstruction_loss + out.l1_loss
                else:
                    dead_mask = (
                        num_frames_since_fired
                        > autoencoder_config["dead_feature_threshold"]
                    )
                    out = dist_model(activations, dead_mask=dead_mask)
                    loss = out.fvu + out.auxk_loss + out.multi_topk_fvu / 8
                    did_fire[out.encoded.top_indices.flatten()] = True
                    num_frames_since_fired += (
                        activations.shape[1] * activations.shape[0]
                    )
                    num_frames_since_fired[did_fire] = 0

            loss.backward()
            torch.nn.utils.clip_grad_norm_(dist_model.parameters(), clip_thresh)
            optimizer.step()
            scheduler.step()

            state["step"] += 1

            pbar.set_postfix({"loss": f"{loss.item():.3f}", "step": state["step"]})

            if isinstance(out, L1ForwardOutput):
                meta["loss_recon"] = out.reconstruction_loss.item()
                meta["loss_l1"] = out.l1_loss.item()
            else:
                meta["fvu"] = out.fvu.item()
                meta["auxk_loss"] = out.auxk_loss.item()
                meta["multi_topk_fvu"] = out.multi_topk_fvu.item()

            if state["step"] % log_tb_every == 0:
                tb_logger.add_scalar("train/loss", loss, state["step"])
                if isinstance(out, L1ForwardOutput):
                    tb_logger.add_scalar(
                        "train/loss_recon", meta["loss_recon"], state["step"]
                    )
                    tb_logger.add_scalar(
                        "train/loss_l1", meta["loss_l1"], state["step"]
                    )
                else:
                    tb_logger.add_scalar("train/fvu", meta["fvu"], state["step"])
                    tb_logger.add_scalar(
                        "train/auxk_loss", meta["auxk_loss"], state["step"]
                    )
                    tb_logger.add_scalar(
                        "train/multi_topk_fvu", meta["multi_topk_fvu"], state["step"]
                    )
                    tb_logger.add_scalar(
                        "train/dead_pct",
                        torch.mean(dead_mask.float()).item(),
                        state["step"],
                    )
                tb_logger.add_scalar(
                    "train/lr", scheduler.get_last_lr()[0], state["step"]
                )

            if state["step"] % save_every == 0:
                save_checkpoint(
                    state, checkpoint_out_dir + "/step" + str(state["step"]) + ".pth"
                )

            if state["step"] % val_every == 0:
                print("Validating...")
                (
                    losses_dict,
                    subbed_transcripts,
                    base_transcripts,
                    base_filenames,
                    encoded_mag_maxes,
                    encoded_mag_stds,
                ) = validate(
                    model,
                    val_folder,
                    device,
                    whisper_config["layer_name"],
                    whisper_config["model"],
                    not logged_base_transcripts,
                    from_disk,
                )
                logged_base_transcripts = True
                if isinstance(model, L1AutoEncoder):
                    print(
                        f"{state['step']} validation, loss_recon={losses_dict['recon']}, loss_l1={losses_dict['l1']}, mse={losses_dict['mse']}"
                    )
                else:
                    print(
                        f"{state['step']} validation, fvu={losses_dict['fvu']}, auxk_loss={losses_dict['auxk_loss']}, multi_topk_fvu={losses_dict['multi_topk_fvu']}, mse={losses_dict['mse']}, dead_pct={torch.mean(dead_mask.float()).item()}"
                    )

                if isinstance(model, L1AutoEncoder):
                    tb_logger.add_scalar(
                        "val/loss_recon", losses_dict["recon"], state["step"]
                    )
                    tb_logger.add_scalar(
                        "val/loss_l1", losses_dict["l1"], state["step"]
                    )
                else:
                    tb_logger.add_scalar("val/fvu", losses_dict["fvu"], state["step"])
                    tb_logger.add_scalar(
                        "val/auxk_loss", losses_dict["auxk_loss"], state["step"]
                    )
                    tb_logger.add_scalar(
                        "val/multi_topk_fvu",
                        losses_dict["multi_topk_fvu"],
                        state["step"],
                    )
                tb_logger.add_scalar("val/mse", losses_dict["mse"], state["step"])

                tb_logger.add_histogram(
                    "val/encoded/magnitude_maxes",
                    np.array(encoded_mag_maxes),
                    state["step"],
                )
                tb_logger.add_histogram(
                    "val/encoded/magnitude_stds",
                    np.array(encoded_mag_stds),
                    state["step"],
                )
                # count number of zero activations
                DEAD_THRESHOLD = 0
                num_dead = np.count_nonzero(encoded_mag_maxes <= DEAD_THRESHOLD)
                tb_logger.add_scalar("val/encoded/num_dead", num_dead, state["step"])
                tb_logger.add_scalar(
                    "val/encoded/percent_dead",
                    num_dead / encoded_mag_maxes.shape[-1],
                    state["step"],
                )

                for i, transcript in enumerate(subbed_transcripts):
                    tb_logger.add_text(
                        f"val/transcripts/reconstructed_{i}", transcript, state["step"]
                    )

                if base_transcripts:
                    for i, transcript in enumerate(base_transcripts):
                        tb_logger.add_text(
                            f"val/transcripts/base_{i}", transcript, state["step"]
                        )
                    for i, filename in enumerate(base_filenames):
                        audio = torchaudio.load(filename)[0]
                        tb_logger.add_audio(
                            f"val/transcripts/audio_{i}",
                            audio,
                            state["step"],
                            sample_rate=16000,
                        )

                save_loss = (
                    losses_dict["recon"]
                    if isinstance(model, L1AutoEncoder)
                    else losses_dict["fvu"]
                )
                if save_loss < state["best_val_loss"]:
                    print("Saving new best validation")
                    state["best_val_loss"] = save_loss
                    save_checkpoint(state, checkpoint_out_dir + "/bestval.pth")
                    pytorch_model_path = model_out[:-3] + ".bestval"
                    torch.save(model, pytorch_model_path)

            if state["step"] >= steps:
                break

        save_checkpoint(
            state, checkpoint_out_dir + "/step" + str(state["step"]) + ".pth"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to train configuration file"
    )
    args = parser.parse_args()
    # load config json
    with open(args.config, "r") as f:
        config = json.load(f)
    config["device"] = torch.device(config["device"])
    train(**config)
