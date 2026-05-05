from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math
from math import cos, pi
import os, typing

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

if __name__ == "__main__":
    simple_training_loop()
