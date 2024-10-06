import torch
import whisper
from torch.amp import autocast
from tqdm import tqdm
from activation_dataset import ActivationDataset
from librispeech_data import get_mels_from_audio_path
from hooked_model import WhisperSubbedActivation
import numpy as np
import random
import os
from autoencoder import AutoEncoder
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


def validate(
    model: torch.nn.Module,
    recon_loss_fn: torch.nn.Module,
    recon_alpha: float,
    val_folder: str,
    device: torch.device,
    activation_dims: int,
    layer_name: str,
    whisper_model: str,
    log_base_transcripts
):
    model.eval()
    whisper_model = whisper.load_model(whisper_model)
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

    val_dataset = ActivationDataset(val_folder, "val")
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False
    )
    encoded_magnitude_values = torch.zeros(
        (len(val_loader), model.n_dict_components)).to(device)
    context_manager = autocast(device_type=str(
        device)) if device == torch.device("cuda") else nullcontext()

    for i, activations in tqdm(enumerate(val_loader), total=len(val_loader)):
        with torch.no_grad(), context_manager:
            activations, filenames = activations
            activations = activations.to(device)
            filenames = filenames[0]
            pred, c = model(activations)
            detached_c = torch.abs(c.detach()).squeeze()
            c_max = torch.max(detached_c, dim=0).values
            encoded_magnitude_values[i] = c_max
            losses_recon.append(
                recon_alpha * recon_loss_fn(pred, activations).item())
            losses_l1.append(torch.norm(
                c, 1, dim=activation_dims).mean().item())
            if i < N_TRANSCRIPTS:
                mels = get_mels_from_audio_path(device, filenames)
                subbed_result = whisper_sub.forward(mels, pred)
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


def mse_loss(input, target, ignored_index, reduction):
    # mse_loss with ignored_index
    mask = target == ignored_index
    out = (input[~mask] - target[~mask]) ** 2
    if reduction == "mean":
        return out.mean()
    elif reduction == "None":
        return out


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
          grad_acc_steps: int,
          clip_thresh: float,
          batch_size: int,
          dl_max_workers: int,
          log_every: int,
          log_tb_every: int,
          save_every: int,
          val_every: int,
          checkpoint: str,
          recon_alpha: float,
          layer_name: str,
          whisper_model: str
          ):
    set_seeds(seed)
    train_dataset = ActivationDataset(train_folder, "train")
    # train_dataset = TokenEmbeddingDataset()
    feat_dim = train_dataset.activation_shape[-1]
    activation_dims = len(train_dataset.activation_shape)
    model = AutoEncoder(feat_dim, n_dict_components).to(device)
    dist_model = model

    # make run dir
    os.makedirs(run_dir, exist_ok=True)
    checkpoint_out_dir = run_dir + "/checkpoints"
    os.makedirs(checkpoint_out_dir, exist_ok=True)

    # setup logging
    tb_logger = prepare_tb_logging(run_dir)
    # add hparams
    hparam_dict = {
        "lr": lr,
        "weight_decay": weight_decay,
        "steps": steps,
        "grad_acc_steps": grad_acc_steps,
        "clip_thresh": clip_thresh,
        "batch_size": batch_size,
        "recon_alpha": recon_alpha,
        "n_dict_components": n_dict_components,
        "layer_name": layer_name,
        "whisper_model": whisper_model,
    }
    tb_logger.add_text("hparams", json.dumps(hparam_dict, indent=4))
    model_out = run_dir + "/model"
    print("Model: %.2fM" % (sum(p.numel()
          for p in model.parameters()) / 1.0e6))
    logged_base_transcripts = False

    optimizer = RAdam(
        dist_model.parameters(), eps=1e-5, lr=lr, weight_decay=weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=steps, eta_min=0)

    dataloader_kwargs = {
        "batch_size": batch_size,
        "pin_memory": False,
        "drop_last": True,
        "num_workers": dl_max_workers,
    }

    train_loader = iter(
        torch.utils.data.DataLoader(
            train_dataset, shuffle=True, **dataloader_kwargs)
    )

    # Object that contains the main state of the train loop
    state = {
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "step": 0,
        "best_val_loss": float("inf"),
        "total_speech_seconds_seen": 0,
        "total_non_speech_seconds_seen": 0,
        "total_time_ms": 0,
    }
    meta = {}
    meta["effective_batch_size"] = batch_size
    meta["model_params"] = sum(x.numel() for x in dist_model.parameters())

    if checkpoint:
        # loading state_dicts in-place
        print(f"Checkpoint: {checkpoint}")
        load_checkpoint(state, checkpoint, device=device)

    recon_loss_fn = partial(mse_loss, ignored_index=-1, reduction="mean")
    total_steps_per_epoch = len(
        train_dataset) // (dataloader_kwargs['batch_size'] * grad_acc_steps)

    epoch = 0
    while True:
        epoch += 1
        print(f"Epoch {epoch}")

        # Initialize tqdm progress bar for this epoch
        pbar = tqdm(total=total_steps_per_epoch,
                    desc=f"Epoch {epoch}", unit="step")

        forward_time = 0
        backward_time = 0
        losses_recon = []
        losses_l1 = []
        step_start_time = time.time()

        for _ in range(total_steps_per_epoch):
            for _ in range(grad_acc_steps):
                try:
                    activations, filenames = next(train_loader)
                    activations = activations.to(device)
                except StopIteration:
                    train_loader = iter(
                        torch.utils.data.DataLoader(
                            train_dataset, shuffle=True, **dataloader_kwargs)
                    )
                    activations, filenames = next(train_loader)
                    activations = activations.to(device)

                # Forward pass
                with autocast(str(device)):
                    start_time = perf_counter()
                    pred, c = dist_model(activations)
                    forward_time += perf_counter() - start_time
                    loss_recon = recon_alpha * recon_loss_fn(pred, activations)
                    loss_l1 = torch.norm(c, 1, dim=activation_dims).mean()
                    loss = loss_recon + loss_l1
                    losses_recon.append(loss_recon.item())
                    losses_l1.append(loss_l1.item())

                    # Backward pass
                    start_time = perf_counter()
                    loss.backward()
                    backward_time += perf_counter() - start_time

            torch.nn.utils.clip_grad_norm_(
                dist_model.parameters(), clip_thresh)
            optimizer.step()
            scheduler.step()
            dist_model.zero_grad()
            state["step"] += 1

            # Update tqdm progress bar
            step_end_time = time.time()
            step_time = step_end_time - step_start_time
            pbar.update(1)
            pbar.set_postfix({
                'loss': f"{loss.item():.3f}",
                'time/step': f"{step_time:.3f}s"
            })
            step_start_time = step_end_time

            meta["loss_recon"] = sum(losses_recon) / grad_acc_steps
            meta["loss_l1"] = sum(losses_l1) / grad_acc_steps
            meta["time_backward"] = backward_time

            # Logging, saving, and validation code (unchanged)
            # ...

            if steps != -1 and state["step"] >= steps:
                pbar.close()
                break

        pbar.close()

        pbar.close()
        
        if steps != -1 and state["step"] >= steps:
            break

    save_checkpoint(state, checkpoint_out_dir +
                    "/step" + str(state["step"]) + ".pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    # load config json
    with open(args.config, "r") as f:
        config = json.load(f)
    config["device"] = torch.device(config["device"])
    train(**config)
