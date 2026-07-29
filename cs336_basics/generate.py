import torch, argparse
from typing import List

from cs336_basics.utils import init_model_from_args, init_rope_from_args, load_checkpoint, init_tokenizer_from_args
from cs336_basics.lm import softmax

def temp_softmax(x: torch.Tensor, temp: float, dim=0):
    """ [vocab] temp -> 0 makes largest x dominates"""
    return softmax(x/temp, dim)

def top_p_sampling(x: torch.Tensor, top_p_threshold: float):
    """ [vocab] sort, then pick largest probs until add to top_p_threshold"""
    sorted_probs, sorted_idx = torch.sort(x, dim=-1, descending=True)
    cumulative_prob = torch.cumsum(sorted_probs, dim=-1)
    mask = cumulative_prob - sorted_probs <= top_p_threshold # Need to exclude current val, create a mask of 11000
    masked_sorted_val = sorted_probs * mask # zero out trailing probs
    x.scatter_(-1, sorted_idx, masked_sorted_val)

@torch.no_grad()    
def generate(model, rope, prompt, max_new_tokens, context_len, temp, top_p_threshold, eot_token) -> List[int]:
    generated_tokens = []
    while len(generated_tokens) < max_new_tokens and len(prompt) + len(generated_tokens) < context_len:
        inputs = torch.concat([torch.tensor(prompt, dtype=torch.long), torch.tensor(generated_tokens, dtype=torch.long)])
        inputs = inputs.unsqueeze(0) # [context_len] -> [1(batch_size), context_len]
        logits = model(inputs, rope) # [token, vocab]
        # print(f"logits.shape={logits.shape}")
        last_token_repr = logits[:,-1,:] # Last token of [1, token, vocab] -> [1, vocab]
        last_token_repr = last_token_repr.squeeze(0) # [1, vocab] -> [vocab]
        # print(f"last_token_repr.shape={last_token_repr.shape}")
        dist = temp_softmax(last_token_repr, temp, dim=-1) # temperatured softmax
        top_p_sampling(dist, top_p_threshold)
        token = torch.multinomial(dist, 1)[0] 
        if token == eot_token:
            return generated_tokens
        generated_tokens.append(token.item())
    return generated_tokens

def parse_args():
    parser = argparse.ArgumentParser(description="Generate from a checkpoint")

    model_args = parser.add_argument_group("model")
    model_args.add_argument("--vocab-size", type=int, required=True)
    model_args.add_argument("--context-length", type=int, default=256)
    model_args.add_argument("--d-model", type=int, default=512)
    model_args.add_argument("--num-layers", type=int, default=4)
    model_args.add_argument("--num-heads", type=int, default=16)
    model_args.add_argument("--d-ff", type=int, default=1344)
    model_args.add_argument("--theta", type=int, default=10000)

    ckpt_args = parser.add_argument_group("checkpointing")
    ckpt_args.add_argument("--checkpoint-path", type=str, required=True, help="Checkpoint file to load model")

    generate_args = parser.add_argument_group("generate")
    generate_args.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    generate_args.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    generate_args.add_argument("--seed", type=int, default=0)
    generate_args.add_argument("--prompt", type=str, required=True, help="Prompt to start generation")
    generate_args.add_argument("--max-new-tokens", type=int, required=True, help="max new token to generate")
    generate_args.add_argument("--temp", type=float, default=0.01)
    generate_args.add_argument("--top-p-threshold", type=float, default=0.8)
    generate_args.add_argument("--eot-str", type=str, default="<|endoftext|>")

    tokenizer_args = parser.add_argument_group("tokenizer")
    tokenizer_args.add_argument("--vocab-file", type=str, required=True, help="Path to the vocabulary JSON file (e.g., tokenizer_output/vocab_tinystories_train.json).")
    tokenizer_args.add_argument("--merges-file", type=str, required=True, help="Path to the merges JSON file (e.g., tokenizer_output/merges_tinystories_train.json).")
    tokenizer_args.add_argument("--special-tokens", nargs="+", default=["<|endoftext|>"], help="List of special tokens to include in the tokenizer.")
    
    return parser.parse_args()

def main(args):
    device = args.device
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(args.seed)

    model = init_model_from_args(args, device, dtype)
    rope = init_rope_from_args(args, device)
    load_checkpoint(args.checkpoint_path, model, device)
    model.eval()

    tokenizer = init_tokenizer_from_args(args)
    eot_token = tokenizer.encode(args.eot_str)[0]

    ids = list(tokenizer.encode_iterable(args.prompt))
    generated_tokens = generate(model, rope, ids, args.max_new_tokens, args.context_length, args.temp, args.top_p_threshold, eot_token)
    print(generated_tokens)
    generated_text = tokenizer.decode(generated_tokens)
    print(args.prompt + generated_text)

if __name__ == "__main__":
    main(parse_args())
    