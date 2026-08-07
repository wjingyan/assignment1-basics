from typing import Tuple
import os, typing, torch

from cs336_basics.lm import TransformerLM, RoPE
from cs336_basics.optimizer import AdamW
from cs336_basics.tokenizer import Tokenizer

def bytes_to_readable(b: Tuple[bytes, ...]) -> str:
    """
    Convert a tuple of bytes to readable format.
    
    Examples:
        (b'm', b'o', b're') → 'm|o|re'
        (b'r', b'e') → 'r|e'
        (b' ', b'a') → ' |a'
    """
    def decode_single(byte_val: bytes) -> str:
        try:
            decoded = byte_val.decode('utf-8')
            if decoded == '\n': return '<newline>'
            elif decoded == '\t': return '<tab>'
            elif decoded == '\r': return '<return>'
            else: return decoded
        except UnicodeDecodeError:
            return f"0x{byte_val.hex()}"
    
    return "|".join(decode_single(b_val) for b_val in b)


def format_byte_pairs(byte_pair_freqs: dict, top_n: int = 10) -> str:
    """
    Format byte pair frequencies in a readable way.
    
    Args:
        byte_pair_freqs: Dictionary mapping byte pairs to frequencies
        top_n: Number of top pairs to display
    
    Returns:
        Formatted string with top byte pairs
    """
    if not byte_pair_freqs:
        return "No byte pairs"
    
    top_pairs = sorted(byte_pair_freqs.items(), key=lambda x: (-x[1], x[0]))[:top_n]
    lines = []
    
    for pair, count in top_pairs:
        left = bytes_to_readable((pair[0],))
        right = bytes_to_readable((pair[1],))
        readable = f"{left}|{right}"
        lines.append(f"  {readable}: {count}")
    
    return "\n".join(lines)

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

def init_rope_from_args(args, device):
    return RoPE(
        theta = args.theta,
        d_k = args.d_model // args.num_heads,
        max_seq_len = args.context_length,
        device = device
    )

def init_tokenizer_from_args(args):
    # Validate paths
    if not os.path.exists(args.vocab_file):
        sys.exit(f"Error: Vocab file not found at {args.vocab_file}")
    if not os.path.exists(args.merges_file):
        sys.exit(f"Error: Merges file not found at {args.merges_file}")

    print(f"Loading tokenizer from:\n  Vocab: {args.vocab_file}\n  Merges: {args.merges_file}")
    tokenizer = Tokenizer.from_files(args.vocab_file, args.merges_file, args.special_tokens)
    return tokenizer

def load_checkpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes], model: torch.nn.Module, device, optimizer: torch.optim.Optimizer | None = None):
    checkpoint = torch.load(src, map_location=device)
    # print(checkpoint)
    # print(f"model.lm_head.weight.mean() before ={model.lm_head.weight.mean()}")
    # Strip the "_orig_mod." prefix that torch.compile adds to state_dict keys,
    # so checkpoints saved from a compiled model load into an uncompiled one.
    model_state = checkpoint["model_state"]
    orig_model_state = {k.replace("_orig_mod.", ""): v for k, v in model_state.items()}
    model.load_state_dict(orig_model_state)
    # print(f"model.lm_head.weight.mean() after ={model.lm_head.weight.mean()}")
    # print(model)
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint["iteration"]

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
    model_state = model._orig_mod.state_dict() if model._orig_mod else model.state_dict()
    optimizer_state = optimizer.state_dict()
    checkpoint = {
        "model_state": model_state,
        "optimizer_state": optimizer_state,
        "iteration": iteration,
    }
    torch.save(checkpoint, out)
