import regex as re
from collections import defaultdict
from typing import Tuple, Union, Iterable, Iterator
import multiprocessing
import os
import time
import json
import psutil
from cs336_basics.pretokenization_example import find_chunk_boundaries

class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.token2id = {token: token_id for token_id, token in vocab.items()}
    
    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | os.PathLike,
        merges_filepath: str | os.PathLike,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        with open(vocab_filepath, "r") as f:
            raw_vocab = json.load(f)
        with open(merges_filepath, "r") as f:
            raw_merges = json.load(f)

        # Convert JSON strings back to internal types (int, bytes)
        vocab = {int(k): v.encode('utf-8') for k, v in raw_vocab.items()}
        merges = [(m[0].encode('utf-8'), m[1].encode('utf-8')) for m in raw_merges]

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        res = []
        if not self.special_tokens:
            chunks = [text]
        else:
            pattern = "|".join(re.escape(token) for token in self.special_tokens)
            # Use capturing groups to keep special tokens in the result
            chunks = re.split(f"({pattern})", text)
            chunks = [c for c in chunks if c]

        for chunk in chunks:
            if chunk in self.special_tokens:
                chunk_bytes = chunk.encode('utf-8')
                if chunk_bytes in self.token2id:
                    res.append(self.token2id[chunk_bytes])
                continue

            for match in re.finditer(PAT, chunk):
                # Convert the pre-token to bytes immediately to handle multi-byte chars correctly
                word_bytes = match.group(0).encode('utf-8')
                i = 0
                while i < len(word_bytes):
                    # Greedily match the longest possible byte sequence in our vocabulary
                    for j in range(len(word_bytes), i, -1):
                        token_bytes = word_bytes[i:j]
                        if token_bytes in self.token2id:
                            res.append(self.token2id[token_bytes])
                            i = j
                            break
        return res
                
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        # Concatenate all byte segments before decoding to handle multi-byte characters
        byte_segments = [self.vocab.get(idx, b"") for idx in ids]
        return b"".join(byte_segments).decode('utf-8', errors='replace')

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


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

    # Example for debugging:
    # input_path = "data/debug.txt"
    # vocab_size = 261

    vocab = {}
    merges = []
    # Initialize vocabulary with 256 byte tokens and special tokens
    next_token_id = 0
    for special_token in special_tokens:
        vocab[next_token_id] = special_token.encode('utf-8')
        next_token_id += 1
    # Add byte tokens
    for i in range(256):
        vocab[next_token_id] = bytes([i])
        next_token_id += 1

    # Step 1 Pre-tokenization build word frequency table word_freqs
    pattern = "|".join(re.escape(token) for token in special_tokens)
    
    pretok_start_time = time.time()

    word_freqs = defaultdict(int)
    num_processes = multiprocessing.cpu_count()
    print(f"input_path: {input_path}, using {num_processes} processes for pre-tokenization")
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    
    pool_args = [(input_path, boundaries[i], boundaries[i+1], pattern) for i in range(len(boundaries)-1)]

    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.starmap(_process_chunk, pool_args)

    # Merge word_freqs from all subprocesses
    for local_freqs in results:
        for word_bytes, count in local_freqs.items():
            word_freqs[word_bytes] += count

    pretok_end_time = time.time()
    print(f"Pre-tokenization took {pretok_end_time - pretok_start_time:.2f} seconds.")

    merge_start_time = time.time()

    # Initialize byte_pair_freqs
    byte_pair_freqs = defaultdict(int)
    for word, freq in word_freqs.items():
        for i in range(len(word) - 1):
            byte_pair = (word[i], word[i + 1])
            byte_pair_freqs[byte_pair] += freq

    # Build inverted index: token -> set of words containing it
    token_to_words = defaultdict(set)
    for word in word_freqs:
        for token in word:
            token_to_words[token].add(word)

    # Step 2: Merge
    while next_token_id < vocab_size:
        # Find the most frequent byte pair, lexicographically greater pair if tied
        if not byte_pair_freqs:
            print("No more byte pairs to merge.") 
            break
        most_freq_pair = max(byte_pair_freqs, key=lambda x: (byte_pair_freqs[x], x))
        
        if byte_pair_freqs[most_freq_pair] <= 0:
            print("No more byte pairs to merge.") 
            break

        new_token = most_freq_pair[0] + most_freq_pair[1]
        vocab[next_token_id] = new_token
        merges.append(most_freq_pair)
        # print(f"Merging {most_freq_pair} into {new_token}")
        # Remove the merged pair from counts
        del byte_pair_freqs[most_freq_pair]
        
        mp0, mp1 = most_freq_pair
        
        # Use word_freqs_to_remove, word_freqs_to_add to track changes
        words_to_remove = []
        words_to_add = defaultdict(int)

        # Optimization: Use inverted index to only iterate over words that actually contain the pair
        if mp0 == mp1:
            words_to_check = token_to_words[mp0]
        else:
            words_to_check = token_to_words[mp0].intersection(token_to_words[mp1])
        
        # We must iterate over a copy or list because we will modify token_to_words in the loop
        for word in list(words_to_check):
            freq = word_freqs[word]
            # Scan for the pair and build new word
            i = 0
            mp_indices = set()
            changed = False
            word_len = len(word)
            
            while i < word_len - 1:
                if word[i] == mp0 and word[i+1] == mp1:
                    mp_indices.add(i)
                    changed = True
                    i += 2
                else:
                    i += 1
            
            if changed:
                # Based on mp_indices, 1. rebuild word with merged token 2. update byte_pair_freqs
                new_word = []
                last_mp_idx = 0
                for j in sorted(mp_indices):
                    new_word.extend(word[last_mp_idx:j])
                    new_word.append(new_token)
                    last_mp_idx = j + 2
                    if j > 0:
                        byte_pair_freqs[(word[j - 1], word[j])] -= freq
                        byte_pair_freqs[(word[j - 1], new_token)] += freq
                    if j < word_len - 2:
                        byte_pair_freqs[(word[j + 1], word[j + 2])] -= freq
                        byte_pair_freqs[(new_token, word[j + 2])] += freq    
                new_word.extend(word[last_mp_idx:]) # Last piece of word

                words_to_remove.append(word)
                words_to_add[tuple(new_word)] += freq
        
        # Apply updates to word_freqs and inverted index
        for word in words_to_remove:
            del word_freqs[word]
            for token in word:
                token_to_words[token].discard(word)

        for new_word, freq in words_to_add.items():
            word_freqs[new_word] += freq
            for token in new_word:
                token_to_words[token].add(new_word)
        # print(f" byte_pair_freqs[(b' t', b'h')]: {byte_pair_freqs.get((b' t', b'h'), 0)}")
        # print(f" byte_pair_freqs[(b'h', b'e')]: {byte_pair_freqs.get((b'h', b'e'), 0)}")
  
        next_token_id += 1

    # Time log
    merge_end_time = time.time()
    print(f"Merging took {merge_end_time - merge_start_time:.2f} seconds.")
    print(f"Total train_bpe function time: {merge_end_time - total_start_time:.2f} seconds.")

    return vocab, merges

def run_bpe_training_and_save(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    output_dir: str,
    output_basename: str,
):
    """
    A wrapper function to train a BPE tokenizer and save the results.
    Includes profiling for time and memory.
    """
    overall_start_time = time.time()

    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
    )

    print(f"Trained BPE vocab size: {len(vocab)}")
    print(f"Trained BPE merges count: {len(merges)}")
    
    # Serialize vocabulary and merges to files.
    os.makedirs(output_dir, exist_ok=True)
    
    vocab_filepath = os.path.join(output_dir, f"vocab_{output_basename}.json")
    merges_filepath = os.path.join(output_dir, f"merges_{output_basename}.json")

    # The vocabulary maps integer IDs to bytes. For human-readable JSON serialization,
    # we can decode the bytes using 'latin-1', which provides a one-to-one
    # mapping from byte values 0-255 to the first 256 Unicode code points.
    # This ensures that any byte sequence can be represented as a string.
    serializable_vocab = {k: v.decode('latin-1') for k, v in vocab.items()}
    
    longest_vocab_token = max(vocab.values(), key=len)
    print(f"Longest vocabulary token: '{longest_vocab_token.decode('latin-1', errors='ignore')}' (length: {len(longest_vocab_token)})")
    
    print(f"Writing vocabulary to {vocab_filepath}...")
    with open(vocab_filepath, "w", encoding="utf-8") as f:
        json.dump(serializable_vocab, f, indent=2)

    # Merges are tuples of bytes. We'll serialize them similarly.
    serializable_merges = [[m[0].decode('latin-1'), m[1].decode('latin-1')] for m in merges]
    print(f"Writing merges to {merges_filepath}...")
    with open(merges_filepath, "w", encoding="utf-8") as f:
        json.dump(serializable_merges, f, indent=2)

    overall_end_time = time.time()
    process = psutil.Process(os.getpid())
    print(f"\n--- Profiling ---")
    print(f"Total execution time: {overall_end_time - overall_start_time:.2f} seconds")
    print(f"Peak memory usage: {process.memory_info().rss / 1024**2:.2f} MB")
    print("For more detailed profiling, run with scalene: `scalene cs336_basics/run_train_bpe.py ...`")