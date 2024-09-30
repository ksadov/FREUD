import torch
from torch.cuda.amp import autocast
from activation_dataset import ActivationDataset
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

def validate(
    model: torch.nn.Module,
    recon_loss_fn: torch.nn.Module,
    recon_alpha: float,
    activation_folder: str,
    device: torch.device,
):
    model.eval()
    losses_recon = []
    losses_l1 = []

    val_dataset = ActivationDataset(activation_folder, "val")
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False
    )
    for activations in val_loader:
        with torch.no_grad() and autocast():
            activations = activations.to(device)
            pred, c = model(activations)
            losses_recon.append(recon_alpha * recon_loss_fn(pred, activations).item())
            losses_l1.append(torch.norm(c, 1, dim=2).mean().item())

    model.train()
    return np.array(losses_recon).mean(), np.array(losses_l1).mean()

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
    device="cpu",
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
          activation_folder: str, 
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
          checkpoint_out: str,
          recon_alpha: float
          ):
    set_seeds(seed)
    train_dataset = ActivationDataset(activation_folder, "train")
    # train_dataset = TokenEmbeddingDataset()
    feat_dim = train_dataset.activation_shape[-1]
    activation_dims = len(train_dataset.activation_shape)
    model = AutoEncoder(feat_dim, n_dict_components).to(device)
    dist_model = model

    # make run dir
    os.makedirs(run_dir, exist_ok=True)

    # setup logging
    tb_logger = prepare_tb_logging(run_dir)
    model_out = run_dir + "/model"
    print("Model: %.2fM" % (sum(p.numel() for p in model.parameters()) / 1.0e6))

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
        torch.utils.data.DataLoader(train_dataset, shuffle=True, **dataloader_kwargs)
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
    while True:
        forward_time = 0
        backward_time = 0
        losses_recon = []
        losses_l1 = []
        for _ in range(grad_acc_steps):
            try:
                activations, filenames = next(train_loader)
                activations = activations.to(device)
            except StopIteration:
                train_loader = iter(
                    torch.utils.data.DataLoader(train_dataset, shuffle=True, **dataloader_kwargs)
                )
                activations, filenames = next(train_loader)
                activations = activations.to(device)
            # Forward pass
            with autocast():
                start_time = perf_counter()
                pred, c = dist_model(activations)  # bsz, seq_len, n_classes
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

        torch.nn.utils.clip_grad_norm_(dist_model.parameters(), clip_thresh)
        optimizer.step()
        scheduler.step()
        dist_model.zero_grad()
        state["step"] += 1
        meta["loss_recon"] = sum(losses_recon) / grad_acc_steps
        meta["loss_l1"] = sum(losses_l1) / grad_acc_steps
        meta["time_backward"] = backward_time

        if state["step"] % log_every == 0:
            print(f"step {state['step']}, loss {loss.item():.3f}")

            # log training losses
            if state["step"] % log_tb_every == 0:
                tb_logger.add_scalar("train/loss", loss, state["step"])
                tb_logger.add_scalar("train/loss_recon", meta["loss_recon"], state["step"])
                tb_logger.add_scalar("train/loss_l1", meta["loss_l1"], state["step"])
                tb_logger.add_scalar("train/lr", scheduler.get_last_lr()[0], state["step"])

        # save out model periodically
        if state["step"] % save_every == 0:
            save_checkpoint(state, checkpoint_out + ".step" + str(state["step"]))

        # validate periodically
        if state["step"] % val_every == 0:
            print("Validating...")
            val_loss_recon, val_loss_l1 = validate(
                model, recon_loss_fn, recon_alpha, activation_folder, device
            )
            print(f"{state['step']} validation, loss_recon={val_loss_recon:.3f}")
            # log validation losses
            tb_logger.add_scalar("val/loss_recon", val_loss_recon, state["step"])
            tb_logger.add_scalar("val/loss_l1", val_loss_l1, state["step"])
            if val_loss_recon.item() < state["best_val_loss"]:
                print("Saving new best validation")
                state["best_val_loss"] = val_loss_recon.item()
                save_checkpoint(state, checkpoint_out + ".bestval")

                # Save PyTorch model for PR area calculation
                pytorch_model_path = model_out[:-3] + ".bestval"
                torch.save(model, pytorch_model_path)

        if steps != -1 and state["step"] >= steps:
            break

    save_checkpoint(state, checkpoint_out + ".step" + str(state["step"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    # load config json
    with open(args.config, "r") as f:
        config = json.load(f)
    train(**config)

   