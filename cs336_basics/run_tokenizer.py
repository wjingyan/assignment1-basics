import argparse
import json
import os
import sys
import time
import numpy as np
from cs336_basics.tokenizer import Tokenizer

def main():
    """
    Command-line interface to load a trained BPE tokenizer and encode text.
    """
    parser = argparse.ArgumentParser(description="Run BPE tokenizer on input text.")
    parser.add_argument(
        "--vocab", 
        type=str, 
        required=True, 
        help="Path to the vocabulary JSON file (e.g., tokenizer_output/vocab_tinystories_train.json)."
    )
    parser.add_argument(
        "--merges", 
        type=str, 
        required=True, 
        help="Path to the merges JSON file (e.g., tokenizer_output/merges_tinystories_train.json)."
    )
    parser.add_argument(
        "--input", 
        type=str, 
        required=True, 
        help="Path to the input text file to encode."
    )
    parser.add_argument(
        "--output", 
        type=str, 
        required=True, 
        help="Path to save the encoded token IDs (NumPy .npy format)."
    )
    parser.add_argument(
        "--special-tokens", 
        nargs="+", 
        default=["<|endoftext|>"], 
        help="List of special tokens to include in the tokenizer."
    )

    args = parser.parse_args()

    # Validate paths
    if not os.path.exists(args.vocab):
        sys.exit(f"Error: Vocab file not found at {args.vocab}")
    if not os.path.exists(args.merges):
        sys.exit(f"Error: Merges file not found at {args.merges}")
    if not os.path.exists(args.input):
        sys.exit(f"Error: Input file not found at {args.input}")

    print(f"Loading tokenizer from:\n  Vocab: {args.vocab}\n  Merges: {args.merges}")
    tokenizer = Tokenizer.from_files(args.vocab, args.merges, args.special_tokens)

    original_bytes = os.path.getsize(args.input)
    print(f"Encoding {args.input} ({original_bytes} bytes) using streaming...")
    
    start_time = time.time()
    with open(args.input, "r", encoding="utf-8") as f:
        ids = list(tokenizer.encode_iterable(f))
    duration = time.time() - start_time
    num_tokens = len(ids)
    print(f"Encoded into {num_tokens} tokens in {duration:.4f} seconds.")

    if num_tokens > 0:
        print(f"Compression Ratio: {original_bytes / num_tokens:.2f} bytes/token")
    if duration > 0:
        print(f"Throughput: {original_bytes / duration:.2f} bytes/second")

    print(f"Serializing {num_tokens} tokens to {args.output} as uint16...")
    ids_array = np.array(ids, dtype=np.uint16)
    np.save(args.output, ids_array)

if __name__ == "__main__":
    main()