import argparse
import os
from cs336_basics.tokenizer import run_bpe_training_and_save

def main():
    """
    Command-line interface to run BPE tokenizer training.
    """
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer.")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["tinystories", "owt", "test"],
        help="The dataset to use for training (tinystories or owt).",
    )
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        choices=["train", "valid"],
        help="The dataset split to use (train or validation).",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=None,
        help="Override the default vocab size (useful for quick profiling runs).",
    )
    args = parser.parse_args()

    # --- Configuration ---
    if args.dataset == "tinystories":
        vocab_size = 10000
        input_path = f"data/TinyStoriesV2-GPT4-{'train' if args.split == 'train' else 'valid'}.txt"
    elif args.dataset == "owt":
        vocab_size = 32000
        input_path = f"data/owt_{args.split}.txt"
    elif args.dataset == "test":
        vocab_size = 10000
        input_path = f"tests/fixtures/corpus.en"
    
    # Override vocab_size if provided
    if args.vocab_size is not None:
        vocab_size = args.vocab_size
    
    special_tokens = ["<|endoftext|>"]
    output_dir = "tokenizer_output"
    output_basename = f"{args.dataset}_{args.split}"

    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        print("Please make sure you have downloaded the datasets as per the assignment instructions.")
        return

    # --- Run Training ---
    run_bpe_training_and_save(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        output_dir=output_dir,
        output_basename=output_basename,
    )

if __name__ == "__main__":
    main()
