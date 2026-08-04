import argparse
import time
import os, typing
import logging
import numpy as np
import torch

from cs336_basics.lm import TransformerLM, cross_entropy, RoPE
from cs336_basics.utils import init_model_from_args, init_optimizer_from_args, init_rope_from_args, load_checkpoint, save_checkpoint
from cs336_basics.optimizer import learning_rate_schedule, gradient_clipping

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

"""
def simple_training_loop():
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=1e3)
    for t in range(10):
        opt.zero_grad() # Reset the gradients for all learnable parameters.
        loss = (weights**2).mean() # Compute a scalar loss value.
        print(loss.cpu().item())
        loss.backward() # Run backward pass, which computes gradients.
        opt.step() # Run optimizer step.
"""

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

@torch.no_grad()
def estimate_val_loss(model, rope, val_data, batch_size, context_length, device, eval_iters):
    model.eval()
    losses = torch.zeros(eval_iters)
    for i in range(eval_iters):
        inputs, targets = load_data(val_data, batch_size, context_length, device)
        logits = model(inputs, rope)
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
    # train_data = np.memmap(args.train_data, dtype=np.uint16, mode="r")
    # val_data = np.memmap(args.val_data, dtype=np.uint16, mode="r")
    train_data = np.load(args.train_data, mmap_mode='r')
    val_data = np.load(args.val_data, mmap_mode='r')

    model = init_model_from_args(args, device, dtype)
    optimizer = init_optimizer_from_args(args, model.parameters())
    rope = init_rope_from_args(args, device)

    # acceleration
    # Compilation with Inductor is not supported on mps as of torch version 2.9.0.
    if device.startswith('mps'):
        model = torch.compile(model, backend="aot_eager")
    else:
        model = torch.compile(model)
    if device.startswith('cuda'):
        torch.set_float32_matmul_precision('high')
        logging.info(f"torch.get_float32_matmul_precision()={torch.get_float32_matmul_precision()}")

    iteration = 0
    if args.resume_from:
        iteration = load_checkpoint(args.resume_from, model, device, optimizer)

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
        logits = model(inputs, rope)
        loss = cross_entropy(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        gradient_clipping(model.parameters(), args.grad_clip)
        optimizer.step()

        if i % args.log_interval == 0:
            elapsed = time.time() - t_start
            logging.info(f"iter {i}: train_loss={loss.item():.4f} lr={lr:.6f} elapsed={elapsed:.1f}s")
            if use_wandb:
                wandb.log({"train/loss": loss.item(), "train/lr": lr, "iter": i}, step=i)

        if i % args.eval_interval == 0 or i == args.max_iters - 1:
            val_loss = estimate_val_loss(model, rope, val_data, args.batch_size, args.context_length, device, args.eval_iters)
            logging.info(f"iter {i}: val_loss={val_loss:.4f}")
            if use_wandb:
                wandb.log({"val/loss": val_loss, "iter": i}, step=i)

        if args.save_interval > 0 and i % args.save_interval == 0 and i > 0:
            save_checkpoint(model, optimizer, i, os.path.join(args.checkpoint_dir, f"ckpt_{i}.pt"))
        
        if elapsed > 2700:
            logging.info("Reached 45 min time limit. Exiting...")
            return

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
    model_args.add_argument("--theta", type=int, default=10000)

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
    ckpt.add_argument("--checkpoint-dir", type=str, required=True, help="Directory to serialize checkpoints to. Use save-interval -1 to skip")
    ckpt.add_argument("--save-interval", type=int, default=500, help="Save a checkpoint every N iterations. -1 for never save")
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
