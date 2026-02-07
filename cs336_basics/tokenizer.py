import regex as re
from collections import defaultdict
from typing import Tuple, Union
import multiprocessing
import os
import time
import json
import psutil
from cs336_basics.pretokenization_example import find_chunk_boundaries

from tests.common import FIXTURES_PATH
 

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


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
        left = bytes_to_readable(pair[0])
        right = bytes_to_readable(pair[1])
        readable = f"{left}|{right}"
        lines.append(f"  {readable}: {count}")
    
    return "\n".join(lines)

def _process_chunk(input_path: str, start: int, end: int, pattern: str) -> dict[tuple[bytes, ...], int]:
    local_word_freqs = defaultdict(int)
    
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk_bytes = f.read(end - start)
        # Decode while ignoring errors to handle potential split characters at boundaries
        # though find_chunk_boundaries should prevent this for special tokens.
        chunk_text = chunk_bytes.decode("utf-8", errors="ignore")
        chunks = re.split(pattern, chunk_text)
        for chunk in chunks:
            for match in re.finditer(PAT, chunk):
                word = match.group(0)
                word_bytes = tuple(bytes([b]) for b in word.encode('utf-8'))
                local_word_freqs[word_bytes] += 1
    return dict(local_word_freqs)

def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
) -> Tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a byte-level BPE tokenizer on raw text data.
    
    Args:
        input_path: Path to text file for training
        vocab_size: Maximum final vocabulary size (including bytes + merges + special tokens)
        special_tokens: Special tokens to add to vocabulary
    Returns:
        vocab: Mapping from token ID to bytes
        merges: Ordered list of (token1, token2) merge operations
    """
    total_start_time = time.time()

    # Debug params
    # input_path = "tests/fixtures/tinystories_sample.txt"
    # input_path = "tests/fixtures/debug.txt"
    # vocab_size = 403

    vocab = {}
    merges = []
    # Initialize vocabulary with 256 byte tokens and special tokens
    # Add special tokens to vocab first
    next_token_id = 0
    for special_token in special_tokens:
        vocab[next_token_id] = special_token.encode('utf-8')
        next_token_id += 1
    # Add byte tokens
    for i in range(256):
        vocab[next_token_id] = bytes([i])
        next_token_id += 1

    
    # with open(input_path, 'r', encoding='utf-8') as f:
    #     text = f.read()
    # Step 1 Pre-tokenization build word frequency table word_freqs
    # Replace special tokens before pre-tokenization with |
    pattern = "|".join(re.escape(token) for token in special_tokens)
    # text = text[:1000]
    # chunks = re.split(pattern, text)
    # print(len(text))
    
    pretok_start_time = time.time()

    word_freqs = defaultdict(int)
    num_processes = multiprocessing.cpu_count()
    print(f"input_path: {input_path}, using {num_processes} processes for pre-tokenization")
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    # TODO Parallelize pre-tokenization
    # for chunk in chunks:
    #     # print(len(chunk))
    #     for match in re.finditer(PAT, chunk):
    #         word = match.group(0)
    #         # print(word)
    #         word_bytes = tuple(bytes([b]) for b in word.encode('utf-8'))
    #         word_freqs[word_bytes] += 1
    
    pool_args = [(input_path, boundaries[i], boundaries[i+1], pattern) for i in range(len(boundaries)-1)]

    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.starmap(_process_chunk, pool_args)

    # Merge word_freqs from all subprocesses
    for local_freqs in results:
        for word_bytes, count in local_freqs.items():
            word_freqs[word_bytes] += count

    pretok_end_time = time.time()
    print(f"Pre-tokenization took {pretok_end_time - pretok_start_time:.2f} seconds.")

    # Print top 10 most frequent words
    # top_10_words = sorted(word_freqs.items(), key=lambda x: -x[1])[:10]
    # for word, freq in top_10_words:
    #     print(f"{word}: {freq}")
    
    merge_start_time = time.time()

    # Step 2: Merge
    while next_token_id < vocab_size:
        # Examine all byte pairs
        byte_pair_freqs = defaultdict(int)
        for word in word_freqs.keys():
            for i in range(len(word) - 1):
                byte_pair = (word[i], word[i + 1])
                byte_pair_freqs[byte_pair] += word_freqs[word]
        # print(f"byte_pair_freqs: {byte_pair_freqs}")
        # space_t = (b' ', b't')
        # print(f"Count of (b' ', b't'): {byte_pair_freqs.get(space_t, 0)}")
        # print(f" byte_pair_freqs[(b' you', b'r')]: {byte_pair_freqs.get((b' you', b'r'), 0)}")
        # print(f" byte_pair_freqs[(b' ', b'1')]: {byte_pair_freqs.get((b' ', b'1'), 0)}")
        
        # Find the most frequent byte pair, lexicographically greater pair if tied
        if not byte_pair_freqs:
            print("No more byte pairs to merge.") 
            break
        most_freq_pair = max(byte_pair_freqs, key=lambda x: (byte_pair_freqs[x], x))
        # print(f"Merging pair {most_freq_pair} for all words at token_id {next_token_id}")
        # if next_token_id == 257+13:
        #     print(f"Most frequent pair at token id {next_token_id}: {most_freq_pair} with count {byte_pair_freqs[most_freq_pair]}")
        # most_freq_pair = max(byte_pair_freqs, key=lambda x: (byte_pair_freqs[x], x[0], x[1]))
        # top_10 = sorted(byte_pair_freqs.items(), key=lambda x: (-x[1], x[0]))[:10]
        # for pair, count in top_10:
        #     print(f"{pair}: {count}")
        # Create new merged token as a single bytes object
        new_token = most_freq_pair[0] + most_freq_pair[1]
        vocab[next_token_id] = new_token
        # Update the word frequencies
        # Change key of word_freqs from (a, new_token[0], new_token[1], b) to (a, new_token, b)
        new_word_freqs = defaultdict(int)
        # debug params
        # ore_cnt = 0
        # o_re_cnt = 0
        # space_1_words = {}  # dict to track words and their frequencies
        # o_re_words = {}
        
        for word in word_freqs.keys():
            word_w_new_token = word
            # debug params
            need_to_merge_0_0 = False
            
            # Find all occurrences of most_freq_pair in word
            i, offset = 0, 0 # offset to account for changes in length after replacements
            while i < len(word) - 1:
                if word[i] == most_freq_pair[0] and word[i + 1] == most_freq_pair[1]:
                    word_w_new_token = word_w_new_token[:(i-offset)] + (new_token,) + word_w_new_token[(i-offset + 2):]
                    offset += 1
                    i += 1  # Move past the newly inserted token
                # if i > 1 and word[i - 2] == b' ' and word[i - 1] == b'1':
                #     space_1_words[bytes_to_readable(word_w_new_token)] = word_freqs[word]
                # if i > len(word) - 1 and word[i] == b'0' and word[i + 1] == b'0':
                #     need_to_merge_0_0 = True
                i += 1
            new_word_freqs[word_w_new_token] = word_freqs[word]

            # if need_to_merge_0_0 and next_token_id == 401:
                # print(f"{bytes_to_readable(word)} -> {bytes_to_readable(word_w_new_token)}")
        
        word_freqs = new_word_freqs

        # if space_1_words and (next_token_id == 400 or next_token_id == 401):
        #     print(f"space_1_words ({len(space_1_words)} total, sorted alphabetically):")
        #     for word_readable in sorted(space_1_words.keys()):
        #         print(f"--{word_readable}: {space_1_words[word_readable]}")

        merges.append(most_freq_pair)
        next_token_id += 1

    merge_end_time = time.time()
    print(f"Merging took {merge_end_time - merge_start_time:.2f} seconds.")

    total_end_time = time.time()
    print(f"Total train_bpe function time: {total_end_time - total_start_time:.2f} seconds.")

    # print(f"vocab: {vocab}")
    # print(f"merges: {merges}")
    return vocab, merges

def train_bpe_tinystories():
    """
    An example function to train a BPE tokenizer on the TinyStories dataset.
    Includes profiling for time and memory, and serializes the trained
    tokenizer's vocabulary to disk.
    """
    overall_start_time = time.time()

    # input_path = "data/TinyStoriesV2-GPT4-train.txt"
    input_path = "data/TinyStoriesV2-GPT4-valid.txt"
    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
    )

    print(f"Trained BPE vocab size: {len(vocab)}")
    print(f"Trained BPE merges count: {len(merges)}")
    
    # Serialize vocabulary to a file.
    # This is done here instead of inside `train_bpe` to avoid changing its
    # function signature, which would break the tests.
    output_dir = "tokenizer"
    os.makedirs(output_dir, exist_ok=True)
    vocab_filepath = os.path.join(output_dir, "vocab_tinystories.json")
    # The vocabulary maps integer IDs to bytes. For human-readable JSON serialization,
    # we can decode the bytes using 'latin-1', which provides a one-to-one
    # mapping from byte values 0-255 to the first 256 Unicode code points.
    # This ensures that any byte sequence can be represented as a string.
    serializable_vocab = {k: v.decode('latin-1') for k, v in vocab.items()}
    # longest_vocab_token = max(len(v) for v in vocab.values())
    longest_vocab_token = max(vocab.values(), key=lambda x: len(x))
    print(f"Longest vocabulary token length: {longest_vocab_token} len(longest_vocab_token): {len(longest_vocab_token)}")
    print(f"Writing vocabulary to {vocab_filepath}...")
    with open(vocab_filepath, "w", encoding="utf-8") as f:
        json.dump(serializable_vocab, f, indent=2)

    overall_end_time = time.time()
    process = psutil.Process(os.getpid())
    print(f"\n--- Profiling ---")
    print(f"Total execution time: {overall_end_time - overall_start_time:.2f} seconds")
    print(f"Peak memory usage: {process.memory_info().rss / 1024**2:.2f} MB")
    print("For more detailed profiling, run with scalene: `scalene cs336_basics/tokenizer.py`")

def train_bpe_expts_owt():
    """
    An example function to train a BPE tokenizer on the TinyStories dataset.
    Includes profiling for time and memory, and serializes the trained
    tokenizer's vocabulary to disk.
    """
    overall_start_time = time.time()

    input_path = "data/owt_train.txt"
    # input_path = "data/owt_valid.txt"
    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=32000,
        special_tokens=["<|endoftext|>"],
    )

    print(f"Trained BPE vocab size: {len(vocab)}")
    print(f"Trained BPE merges count: {len(merges)}")
    
    # Serialize vocabulary to a file.
    # This is done here instead of inside `train_bpe` to avoid changing its
    # function signature, which would break the tests.
    output_dir = "tokenizer"
    os.makedirs(output_dir, exist_ok=True)
    vocab_filepath = os.path.join(output_dir, "vocab_owt.json")
    # The vocabulary maps integer IDs to bytes. For human-readable JSON serialization,
    # we can decode the bytes using 'latin-1', which provides a one-to-one
    # mapping from byte values 0-255 to the first 256 Unicode code points.
    # This ensures that any byte sequence can be represented as a string.
    serializable_vocab = {k: v.decode('latin-1') for k, v in vocab.items()}
    # longest_vocab_token = max(len(v) for v in vocab.values())
    longest_vocab_token = max(vocab.values(), key=lambda x: len(x))
    print(f"Longest vocabulary token length: {longest_vocab_token} len(longest_vocab_token): {len(longest_vocab_token)}")
    print(f"Writing vocabulary to {vocab_filepath}...")
    with open(vocab_filepath, "w", encoding="utf-8") as f:
        json.dump(serializable_vocab, f, indent=2)

    overall_end_time = time.time()
    process = psutil.Process(os.getpid())
    print(f"\n--- Profiling ---")
    print(f"Total execution time: {overall_end_time - overall_start_time:.2f} seconds")
    print(f"Peak memory usage: {process.memory_info().rss / 1024**2:.2f} MB")
    print("For more detailed profiling, run with scalene: `scalene cs336_basics/tokenizer.py`")

if __name__ == "__main__":
    # train_bpe_tinystories()
    train_bpe_expts_owt()