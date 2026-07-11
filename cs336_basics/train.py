from collections.abc import Callable, Iterable
from typing import Optional
import argparse
import time
import torch
import math
from math import cos, pi
import os, typing
import numpy as np

from cs336_basics.lm import TransformerLM, cross_entropy

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p.
                t = state.get("t", 0) # Get iteration number from the state, or initial value.
                grad = p.grad.data # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place.
                state["t"] = t + 1 # Increment iteration number.
        return loss

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, weight_decay, betas=(0.9, 0.999), eps=1e-8):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay} # group params
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"][0], group["betas"][1]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                m = state.get("m", 0)
                v = state.get("v", 0)
                t = state.get("t", 1)
                m = beta1 * m + (1 - beta1) * p.grad.data
                v = beta2 * v + (1 - beta2) * p.grad.data**2
                lr_t = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)
                p.data -= lr_t * (m / (torch.sqrt(v) + eps)) + lr * weight_decay * p.data
                state["t"] = t + 1
                state["m"] = m
                state["v"] = v
        return loss

def simple_training_loop():
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=1e3)
    for t in range(10):
        opt.zero_grad() # Reset the gradients for all learnable parameters.
        loss = (weights**2).mean() # Compute a scalar loss value.
        print(loss.cpu().item())
        loss.backward() # Run backward pass, which computes gradients.
        opt.step() # Run optimizer step.

def learning_rate_schedule(t, lr_max, lr_min, t_w, t_c):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        t (int): Iteration number to get learning rate for.
        lr_max (float): Maximum learning rate.
        lr_min (float): Minimum learning rate.
        t_w: warm up until t_w
        t_c: cosine annealing until t_c
    """
    if t < t_w:
        return t/t_w * lr_max
    elif t <= t_c:
        return lr_min + 0.5 * (1 + cos((t-t_w)*pi/(t_c-t_w)))*(lr_max-lr_min)
    else: # t > t_c
        return lr_min

def gradient_clipping(params: Iterable[torch.nn.Parameter], max_l2_norm: float):
    eps = 1e-6
    params = list(params)
    total_norm = torch.sqrt(sum(p.grad.data.pow(2).sum() for p in params if p.grad is not None))
    if total_norm > max_l2_norm:
        for p in params:
            if not p.grad is None:
                p.grad.data.mul_(max_l2_norm/(total_norm + eps))

def load_data(x, batch_size, context_length, device):
    """
    Sample random batches from a token dataset.

    Each sample is a contiguous slice of context_length tokens. The target is
    the same slice shifted right by 1 (next-token prediction). Start positions
    are sampled uniformly so that x[i : i+context_length+1] never goes out of bounds.

    Args:
        x (np.array): 1D integer array of token IDs.
        batch_size (int): Number of sequences to sample.
        context_length (int): Length of each sampled sequence.
        device (str): PyTorch device string, e.g. 'cpu' or 'cuda:0'.

    Returns:
        Tuple of LongTensors (inputs, targets), each shape (batch_size, context_length).
    """
    # start_indices: batch_size random positions in [0, len(x)-context_length)
    # upper bound ensures x[i+context_length] exists for the target's last token
    start_indices = torch.randint(0, len(x) - context_length, (batch_size,))
    inputs = torch.stack([torch.tensor(x[i:i+context_length], dtype=torch.long) for i in start_indices])
    targets = torch.stack([torch.tensor(x[i+1:i+context_length+1], dtype=torch.long) for i in start_indices])
    return inputs.to(device), targets.to(device)

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
    model_state = model.state_dict()
    optimizer_state = optimizer.state_dict()
    checkpoint = {
        "model_state": model_state,
        "optimizer_state": optimizer_state,
        "iteration": iteration,
    }
    torch.save(checkpoint, out)

def load_checkpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes], model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint["iteration"]

def init_model_from_args(args, device, dtype):
    return TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        device=device,
        dtype=dtype,
    )

def init_optimizer_from_args(args, params):
    return AdamW(
        params,
        lr=args.lr_max,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
    )

@torch.no_grad()
def estimate_val_loss(model, val_data, batch_size, context_length, device, eval_iters):
    model.eval()
    losses = torch.zeros(eval_iters)
    for i in range(eval_iters):
        inputs, targets = load_data(val_data, batch_size, context_length, device)
        logits = model(inputs)
        losses[i] = cross_entropy(logits, targets).item()
    model.train()
    return losses.mean().item()

def do_train(args):
    device = args.device
    dtype = getattr(torch, args.dtype)

    torch.manual_seed(args.seed)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Memory-efficient loading: the underlying token arrays are memory-mapped
    # from disk, so only the sampled batches are ever materialized in RAM.
    train_data = np.memmap(args.train_data, dtype=np.uint16, mode="r")
    val_data = np.memmap(args.val_data, dtype=np.uint16, mode="r")

    model = init_model_from_args(args, device, dtype)
    optimizer = init_optimizer_from_args(args, model.parameters())

    iteration = 0
    if args.resume_from:
        iteration = load_checkpoint(args.resume_from, model, optimizer)

    use_wandb = args.wandb_project is not None
    if use_wandb:
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    t_start = time.time()
    for i in range(iteration, args.max_iters):
        lr = learning_rate_schedule(i, args.lr_max, args.lr_min, args.warmup_iters, args.cosine_cycle_iters)
        for group in optimizer.param_groups:
            group["lr"] = lr

        inputs, targets = load_data(train_data, args.batch_size, args.context_length, device)
        logits = model(inputs)
        loss = cross_entropy(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        gradient_clipping(model.parameters(), args.grad_clip)
        optimizer.step()

        if i % args.log_interval == 0:
            elapsed = time.time() - t_start
            print(f"iter {i}: train_loss={loss.item():.4f} lr={lr:.6f} elapsed={elapsed:.1f}s")
            if use_wandb:
                wandb.log({"train/loss": loss.item(), "train/lr": lr, "iter": i}, step=i)

        if i % args.eval_interval == 0 or i == args.max_iters - 1:
            val_loss = estimate_val_loss(model, val_data, args.batch_size, args.context_length, device, args.eval_iters)
            print(f"iter {i}: val_loss={val_loss:.4f}")
            if use_wandb:
                wandb.log({"val/loss": val_loss, "iter": i}, step=i)

        if i % args.save_interval == 0 and i > 0:
            save_checkpoint(model, optimizer, i, os.path.join(args.checkpoint_dir, f"ckpt_{i}.pt"))

    save_checkpoint(model, optimizer, args.max_iters, os.path.join(args.checkpoint_dir, "ckpt_final.pt"))
    if use_wandb:
        wandb.finish()

def parse_args():
    parser = argparse.ArgumentParser(description="Train a Transformer language model.")

    data = parser.add_argument_group("data")
    data.add_argument("--train-data", type=str, required=True, help="Path to a .npy/.bin file of uint16 token ids used for training (loaded via np.memmap).")
    data.add_argument("--val-data", type=str, required=True, help="Path to a .npy/.bin file of uint16 token ids used for validation (loaded via np.memmap).")

    model_args = parser.add_argument_group("model")
    model_args.add_argument("--vocab-size", type=int, required=True)
    model_args.add_argument("--context-length", type=int, default=256)
    model_args.add_argument("--d-model", type=int, default=512)
    model_args.add_argument("--num-layers", type=int, default=4)
    model_args.add_argument("--num-heads", type=int, default=16)
    model_args.add_argument("--d-ff", type=int, default=1344)

    opt = parser.add_argument_group("optimizer")
    opt.add_argument("--lr-max", type=float, default=3e-4)
    opt.add_argument("--lr-min", type=float, default=3e-5)
    opt.add_argument("--warmup-iters", type=int, default=100)
    opt.add_argument("--cosine-cycle-iters", type=int, default=1000)
    opt.add_argument("--weight-decay", type=float, default=0.01)
    opt.add_argument("--beta1", type=float, default=0.9)
    opt.add_argument("--beta2", type=float, default=0.999)
    opt.add_argument("--eps", type=float, default=1e-8)
    opt.add_argument("--grad-clip", type=float, default=1.0)

    train_args = parser.add_argument_group("training")
    train_args.add_argument("--batch-size", type=int, default=32)
    train_args.add_argument("--max-iters", type=int, default=5000)
    train_args.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    train_args.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    train_args.add_argument("--seed", type=int, default=0)

    ckpt = parser.add_argument_group("checkpointing")
    ckpt.add_argument("--checkpoint-dir", type=str, required=True, help="Directory to serialize checkpoints to.")
    ckpt.add_argument("--save-interval", type=int, default=500, help="Save a checkpoint every N iterations.")
    ckpt.add_argument("--resume-from", type=str, default=None, help="Path to a checkpoint to resume training from.")

    logging_args = parser.add_argument_group("logging")
    logging_args.add_argument("--log-interval", type=int, default=10, help="Log training loss to console every N iterations.")
    logging_args.add_argument("--eval-interval", type=int, default=200, help="Compute validation loss every N iterations.")
    logging_args.add_argument("--eval-iters", type=int, default=50, help="Number of batches to average for a validation loss estimate.")
    logging_args.add_argument("--wandb-project", type=str, default=None, help="If set, log metrics to this Weights & Biases project.")
    logging_args.add_argument("--wandb-run-name", type=str, default=None)

    return parser.parse_args()

if __name__ == "__main__":
    do_train(parse_args())
