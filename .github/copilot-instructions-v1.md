# CS336 Assignment 1: Basics - AI Copilot Instructions

## Project Overview
This is a Stanford CS336 assignment implementing a **Byte-Pair Encoding (BPE) tokenizer** and a complete **Transformer language model** from scratch using PyTorch. The project uses **`uv`** for dependency management and follows a strict **adapter pattern** separating test infrastructure from implementation code.

## Critical Architecture: Adapter Pattern

**The adapter pattern MUST be followed strictly:**

```
Test → tests/adapters.py (glue code only) → Your implementation
test_model.py → run_linear() → linear implementation in ans.py/cs336_basics/
test_train_bpe.py → run_train_bpe() → train_bpe() in ans.py
```

### Rules
1. **NEVER implement logic in `tests/adapters.py`** - adapters are 5-line glue code only
2. **Implement ALL substantial logic in:**
   - `cs336_basics/` directory for transformer components
3. **Every adapter MUST have detailed docstrings** with:
   - Exact tensor shapes using jaxtyping notation: `Float[Tensor, "batch seq d_model"]`
   - What to compute (e.g., "softmax(QK^T/sqrt(d_k))V")
   - Shape transformations and special handling (3D/4D tensors, causal masking)
   - Return type and semantics

### Anti-Pattern Example (❌ WRONG)
```python
# In adapters.py - DO NOT DO THIS
def run_linear(...):
    result = in_features @ weights.T  # ❌ Implements logic in adapter
    return result
```

### Correct Pattern (✅ RIGHT)
```python
# In adapters.py - CORRECT
def run_linear(d_in, d_out, weights, in_features):
    from cs336_basics.linear import Linear  # Import your implementation
    return Linear(weights)(in_features)  # Just call it

# In cs336_basics/linear.py - ACTUAL IMPLEMENTATION
class Linear:
    def __init__(self, weights):
        self.weights = weights
    def __call__(self, x):
        return x @ self.weights.T
```

## Quick Start & Testing

### Commands
```bash
# Run all tests
uv run pytest

# Run specific test suite
uv run pytest tests/test_model.py           # Transformer components
uv run pytest tests/test_train_bpe.py       # BPE tokenizer
uv run pytest tests/test_nn_utils.py        # Loss/softmax/gradient clipping
uv run pytest tests/test_optimizer.py       # AdamW and LR schedules
uv run pytest tests/test_data.py            # Batch sampling
uv run pytest tests/test_serialization.py   # Checkpointing

# Run single test with debugging
uv run pytest tests/test_model.py::test_linear -v -s
```

### Snapshot-Based Testing
- Tests compare outputs to `.npz` files in `tests/_snapshots/`
- Tolerance is test-specific: `atol=1e-6` for attention, `atol=1e-5` for SwiGLU
- If output matches within tolerance, test passes
- Snapshots validate correctness against reference implementation

## Dependency & Environment Setup

| Tool | Purpose | Notes |
|------|---------|-------|
| **uv** | Dependency/environment management | Run all code via `uv run` |
| **torch** | Tensor operations | Intel Mac: 2.2.2, others: 2.6.0 |
| **regex** | GPT-2 tokenization regex | More powerful than builtin `re` |
| **jaxtyping** | Tensor type hints | Runtime shape validation |
| **einops** | Tensor reshaping | Use for dimension manipulation |

### Low-Resource Tip
- Apple M3 Max: ~5 min on Metal GPU (MPS), ~30 min on CPU
- Use `device='mps'` or `device='cpu'` in training scripts
- Downscale datasets/models for faster iteration

## Implementation Groups by Complexity

This assignment has **distinct implementation tracks** grouped by difficulty and interdependencies:

### Track 1: BPE Tokenizer (Foundation - 30+ points)
**Module:** `train_bpe.py` (create in cs336_basics/)
**Adapters:** `run_train_bpe()` in `tests/adapters.py`
**Tests:** `test_train_bpe.py` (3 tests: speed, correctness, special tokens)

#### 1. **`train_bpe(input_path, vocab_size, special_tokens) → (vocab, merges)`** (15 points)

**Implementation Strategy:**

The BPE algorithm has 3 phases: **Initialization → Pre-tokenization → Iterative Merging**

**Phase 1: Initialization**
```python
# Start with 256 byte tokens (0-255)
vocab = {i: bytes([i]) for i in range(256)}

# Add special tokens as fixed token IDs (starting after 256)
for i, token in enumerate(special_tokens):
    vocab[256 + i] = token.encode('utf-8')

num_merges = vocab_size - len(vocab)  # How many merges to perform
```

**Phase 2: Pre-tokenization**
- **GPT-2 regex pattern**: `r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""`
- **CRITICAL:** Use `regex` module (not `re`!) - only `regex` supports `\p{L}` Unicode classes
- Split corpus on special tokens FIRST to prevent merges across boundaries
  ```python
  import regex
  special_tokens_pattern = "|".join(regex.escape(st) for st in special_tokens)
  chunks = regex.split(f'({special_tokens_pattern})', corpus_text)
  # chunks = ['Doc1', '<|endoftext|>', 'Doc2', '<|endoftext|>', ...]
  ```
- For each non-special chunk, use `regex.finditer()` to extract pre-tokens
- Represent each pre-token as sequence of bytes: `[b't', b'h', b'e']` for "the"
- Build frequency dict: `{(b't', b'h'): 5, (b'h', b'e'): 5, ...}` by counting adjacent pairs

**Phase 3: Iterative Merging (Performance Critical!)**
```python
merges = []
for merge_step in range(num_merges):
    # Find most frequent pair
    max_pair = max(pair_counts, key=pair_counts.get)
    if pair_counts[max_pair] == 0:
        break  # No more pairs to merge
    
    # Create new token ID for this merged pair
    new_token_id = len(vocab)
    merged_bytes = max_pair[0] + max_pair[1]
    vocab[new_token_id] = merged_bytes
    merges.append(max_pair)
    
    # Update pair counts: only update pairs overlapping the merge
    # Don't recount ALL pairs - this kills performance!
    update_pair_counts(pair_counts, max_pair, new_token_id, ...)
```

**Optimization: Incremental Pair Counting (ESSENTIAL for <1.5s speed)**
- After each merge, only update pair counts that involved the old tokens
- Example: if merge `(b'a', b'b')` → `ab`, update counts for:
  - Pairs ending in `a`: `(..., a)` → `(..., ab)` 
  - Pairs starting with `b`: `(b, ...)` → `(ab, ...)`
  - Don't recount all 256² possible pairs!

**Return Format:**
```python
vocab = {
    0: b'\x00',          # byte 0
    1: b'\x01',          # byte 1
    ...
    255: b'\xff',        # byte 255
    256: b'<|endoftext|>', # special token
    257: b'the',         # first merge
    258: b' the',        # second merge
    ...
}

merges = [
    (b't', b'h'),        # first merge: t+h → 'th'
    (b'th', b'e'),       # second merge: th+e → 'the'
    (b' ', b'the'),      # third merge: ' '+the → ' the'
    ...
]
```

**Tests:**
- `test_train_bpe_speed`: Must complete in < 1.5 seconds (reference does 0.38s)
- `test_train_bpe`: Merges must match reference implementation exactly
- `test_train_bpe_special_tokens`: Special tokens never merged, no `<|` substring in other tokens

**Common Implementation Mistakes:**
1. ❌ Using `re` instead of `regex` → GPT-2 pattern fails silently
2. ❌ Merging across special token boundaries → `test_train_bpe_special_tokens` fails
3. ❌ Naive O(pair_space) iteration each merge → timeout on corpus
4. ❌ Not breaking ties with `max()` → lexicographically wrong merges
5. ❌ Modifying pair counts in-place with wrong logic → diverges from reference

---

#### 2. **`Tokenizer` class** (15 points - NEXT AFTER train_bpe)
- Methods: `encode(text) → List[int]`, `decode(ids) → str`, `from_files()` classmethod, `encode_iterable()`
- Pre-tokenize with GPT-2 regex: `r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""`
- Apply merges in order within each pre-token (no cross-boundary merging)
- Handle special tokens (preserve as single tokens)
- **Memory efficiency:** Implement `encode_iterable()` for streaming large files without loading all to RAM
- **Error handling:** Use `errors='replace'` for invalid UTF-8 sequences

### Track 2: Core Model Layers (1-2 points each, foundation for higher-level)
**Module:** `cs336_basics/` (create submodules as needed)
**Adapters:** Individual `run_*()` functions
**Tests:** Snapshot-based validation in `test_model.py`

**Must implement in order (dependencies):**

1. **`Linear` class** (1 point) - `run_linear()`
   - Inherit from `nn.Module`, use `nn.Parameter` for weight matrix W
   - Store W as `(d_out, d_in)` for row-major efficiency
   - Forward: `output = input @ W.T`
   - Initialize with `trunc_normal_` using σ² = 2/(d_in + d_out)

2. **`Embedding` class** (1 point) - `run_embedding()`
   - Forward: `output = weights[token_ids]` (simple indexing)
   - Initialize embedding matrix with `trunc_normal_` using σ² = 1

3. **`RMSNorm` class** (1 point) - `run_rmsnorm()`
   - Formula: `weight * x / sqrt(mean(x²) + eps)`
   - **CRITICAL:** Upcast to float32 before computing, downcast result to original dtype
   - Initialize weight to 1.0, use eps=1e-5 default

4. **Activation functions** (1-2 points total)
   - `run_silu()`: `x * sigmoid(x)` (element-wise)
   - `run_swiglu()`: `(w1 @ x) ⊙ (w3 @ x) @ w2` where ⊙ is element-wise multiply

5. **RoPE rotary embeddings** (2 points) - `run_rope()`
   - Apply 2D rotation matrices based on token position m
   - For each dim pair 2j: rotate by angle m * θ^(-2j/d_k)
   - Works with arbitrary batch dimensions

### Track 3: Attention Mechanisms (5+ points each, core model)
**Adapters:** `run_scaled_dot_product_attention()`, `run_multihead_self_attention()`, `run_multihead_self_attention_with_rope()`
**Tests:** 1e-6 tolerance snapshots

1. **`scaled_dot_product_attention()` (5 points) - `run_scaled_dot_product_attention()`
   - Core formula: `softmax(QK^T / sqrt(d_k) + mask) @ V`
   - **Mask semantics:** True = attend, False = mask (set to -inf before softmax)
   - **Key:** Apply mask BEFORE softmax, AFTER scaling
   - **Stability:** Use max-subtraction in softmax
   - Handles 3D `(batch, seq, d)` and 4D `(batch, heads, seq, d_k)` tensors

2. **`multihead_self_attention()` (5 points) - `run_multihead_self_attention()`
   - Project Q,K,V for all heads in single matmul (d_model × d_model matrices)
   - Reshape projected outputs: `(batch, seq, d_model) → (batch, seq, num_heads, d_k)`
   - Apply scaled_dot_product_attention per head
   - Concatenate heads and apply output projection
   - NO causal masking, NO RoPE

3. **`multihead_self_attention_with_rope()` (extends above)** - `run_multihead_self_attention_with_rope()`
   - Same as multihead_self_attention but apply RoPE to Q and K after projection
   - Default token_positions = arange(seq_len) if not provided

### Track 4: Full Model Assembly (3-5 points each, integration)
**Adapters:** `run_transformer_block()`, `run_transformer_lm()`
**Tests:** Snapshot validation

1. **`TransformerBlock` class (3 points) - `run_transformer_block()`
   - **Pre-norm architecture (CRITICAL ORDER):**
     ```
     x_residual = x
     x = RMSNorm(x)
     x = MHA_with_RoPE(x)
     x = x + x_residual
     
     x_residual = x
     x = RMSNorm(x)
     x = SwiGLU(x)
     x = x + x_residual
     ```
   - Weights dict keys: `attn.{q,k,v,output}_proj.weight`, `ln1.weight`, `ln2.weight`, `ffn.{w1,w2,w3}.weight`
   - Input/output shape: `(batch, seq_len, d_model)`

2. **`TransformerLM` class (3 points) - `run_transformer_lm()`
   - Pipeline:
     1. Embed: `token_embeddings[input_ids]` → `(batch, seq, d_model)`
     2. Apply N transformer blocks with RoPE
     3. Final RMSNorm
     4. Project to logits: `x @ lm_head.T` → `(batch, seq, vocab_size)`
   - Handle truncated sequences: seq_len ≤ context_length
   - Weights dict: `token_embeddings.weight`, `layers.{i}.attn.*`, `layers.{i}.ln*`, `layers.{i}.ffn.*`, `ln_final.weight`, `lm_head.weight`

### Track 5: Training Infrastructure (1-3 points each)
**Module:** `cs336_basics/` or implementations in adapters
**Tests:** `test_nn_utils.py`, `test_optimizer.py`, `test_data.py`, `test_serialization.py`

1. **Loss functions:**
   - `run_softmax(x, dim)` (1 point): Numerically stable with max subtraction
   - `run_cross_entropy(logits, targets)` (1 point): `-log(softmax(logits)[targets])`, returns scalar

2. **Optimization:**
   - `get_adamw_cls() → AdamW` (2 points): Optimizer with lr, weight_decay, betas, eps
   - `run_gradient_clipping(params, max_norm)` (1 point): Compute L2 norm, scale gradients in-place
   - `run_get_lr_cosine_schedule(it, max_lr, min_lr, warmup_iters, cosine_iters)` (1 point): Linear warmup then cosine decay

3. **Data/checkpointing:**
   - `run_get_batch(dataset, batch_size, context_length, device)` (2 points): Sample random windows, return (X, Y) where Y = X shifted by 1
   - `run_save_checkpoint(model, optimizer, iteration, path)` (1 point): torch.save() dict
   - `run_load_checkpoint(path, model, optimizer) → iteration` (1 point): torch.load() and restore states

---

## Typical Implementation Order (Dependencies)
1. **`Linear` + `Embedding`** → Basic layers
2. **`RMSNorm` + activation functions** → Normalization
3. **`scaled_dot_product_attention`** → Attention core
4. **`multihead_self_attention`** + **`RoPE`** → Advanced attention
5. **`TransformerBlock`** → Assembly of layers
6. **`TransformerLM`** → Full model
7. **Training utils** (softmax, cross_entropy, optimizer, etc.) → Training loop
8. **`train_bpe` + `Tokenizer`** → Data preprocessing

---

## TRANSFORMER COMPONENTS REFERENCE - Quick Lookup



---

## Key Tensor Shape Conventions

- **Batch dimension:** First (e.g., `batch`)
- **Sequence dimension:** After batch (e.g., `seq_len`, `sequence_length`)
- **Feature dimensions:** Last (e.g., `d_model`, `d_ff`, `vocab_size`)
- **Multi-head packing:** `(d_head * num_heads, ...)` NOT explicit `(batch, heads, ...)`
  - After projection, reshape to separate heads, then compute attention
  - Example: `(batch, seq, d_model) @ (d_model, d_model).T → (batch, seq, d_model)` then reshape

### jaxtyping Notation Examples
```python
Float[Tensor, "batch seq d_model"]        # 3D tensor
Float[Tensor, "batch heads seq d_k"]      # 4D multi-head tensor
Float[Tensor, "... d_model"]              # Arbitrary batch dims + feature
Int[Tensor, " ..."]                       # Arbitrary shape, integer type
Float[Tensor, ""]                         # Scalar tensor
```

---

## Common Implementation Patterns

### Residual Connections
- ALWAYS add: `output = sublayer_output + input`
- Place AFTER sublayer, before next LayerNorm

### Pre-Norm Architecture
```python
# Correct order for transformer block:
x_residual = x
x = LayerNorm(x)
x = MultiHeadAttention(x)
x = x + x_residual

x_residual = x
x = LayerNorm(x)
x = SwiGLU(x)
x = x + x_residual
```

### Attention Causal Masking
```python
# For autoregressive models: prevent attending to future tokens
# Create mask: (seq_len, seq_len) with True for valid positions, False for masked
mask = torch.tril(torch.ones(seq_len, seq_len, dtype=bool))
# Apply BEFORE softmax: set masked positions to -inf
attn_weights = attn_weights.masked_fill(~mask, float('-inf'))
```

### Stable Softmax
```python
# Subtract max to prevent overflow
max_val = x.max(dim=dim, keepdim=True)[0]
exp_x = torch.exp(x - max_val)
softmax_x = exp_x / exp_x.sum(dim=dim, keepdim=True)
```

### Stable RMSNorm
```python
# RMS = sqrt(mean(x²))
rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + eps)
normalized = x / rms
return weight * normalized  # weight is learnable scale
```

---

## Testing & Validation

### Snapshot Files
- Located in `tests/_snapshots/`
- Named: `test_<function_name>.npz` for numpy arrays
- Snapshots contain reference outputs from reference implementation

### Tolerance Values (from tests)
- `test_linear`: uses default `atol=1e-2, rtol=1e-4`
- `test_swiglu`: `atol=1e-5`
- `test_scaled_dot_product_attention`: `atol=1e-6`
- `test_multihead_self_attention`: `atol=1e-6`
- `test_transformer_block`: `atol=1e-6`

### Debugging Failed Tests
1. Check exact tolerance in test function
2. Print both actual and expected shapes
3. Verify jaxtyping annotations catch shape mismatches
4. Use `pytest -v -s` for detailed output
5. Compare numerically against PyTorch reference implementations when available

---

## Project Structure

```
.
├── tests/
│   ├── adapters.py                  # Glue code calling your implementations
│   ├── test_train_bpe.py            # BPE tests
│   ├── test_model.py                # Transformer component tests
│   ├── test_nn_utils.py             # Loss/softmax/clipping tests
│   ├── test_optimizer.py            # AdamW/LR schedule tests
│   ├── conftest.py                  # Fixtures & snapshot utilities
│   ├── _snapshots/                  # Reference outputs
│   └── fixtures/                    # Test data
├── cs336_basics/                    # Your transformer implementations
│   ├── __init__.py
│   └── pretokenization_example.py   # Helper for BPE chunking
├── pyproject.toml                   # Dependencies & uv config
└── .github/
    └── copilot-instructions.md      # This file
```

---

## Workflows & Troubleshooting

### Typical Implementation Workflow
1. Read the adapter function docstring in `tests/adapters.py`
2. Check the test function in `tests/test_*.py` for fixture shapes and tolerances
3. Implement in `cs336_basics/`
4. Update adapter to import and call your implementation (5 lines max)
5. Run test: `uv run pytest tests/test_<name>.py::test_<name> -v -s`
6. Compare output to snapshot tolerance

### Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| `NotImplementedError` | Adapter not calling your implementation; check import in adapter |
| Shape mismatch | Check tensor shape in adapter docstring; use jaxtyping for validation |
| Tolerance failure | Verify numerical stability (exp overflow, division by zero); check eps values |
| BPE special tokens in merges | Filter special tokens during pre-tokenization; split corpus on them first with `regex.split()` |
| Attention giving NaN | Apply mask BEFORE softmax; use stable softmax with max subtraction; check causal mask dimensions |
| RMSNorm numerical issues | Upcast to float32 BEFORE squaring; use sufficient eps (1e-5); use sqrt(mean(x²)) not mean(sqrt(x²)) |
| Weight mismatch in state_dict | Verify state_dict key names match exactly; remember `weight.T` semantics for linear layers |
| RoPE producing wrong rotations | Check formula: angle = position * theta^(-2j/d_k); apply to even/odd dimension pairs correctly |
| Multihead attention shape errors | After projection to (batch, seq, d_model), reshape to (batch, seq, num_heads, d_k) before attention |
| Pre-norm residual error | Residual MUST be added AFTER attention/FFN, not inside - order is: LayerNorm → sublayer → add residual |
---

## Deep Dive: train_bpe() Implementation

This section provides step-by-step guidance for implementing the `train_bpe()` function efficiently and correctly.

### Step-by-Step Algorithm

#### Step 1: Initialize Vocabulary
```python
def train_bpe(input_path, vocab_size, special_tokens):
    # Start with 256 byte tokens
    vocab = {i: bytes([i]) for i in range(256)}
    
    # Add special tokens as fixed IDs (starting at 256)
    for i, token in enumerate(special_tokens):
        vocab[256 + i] = token.encode('utf-8')
    
    num_merges_needed = vocab_size - len(vocab)
```

**Key points:**
- Vocab must map token_id (int) → token_bytes (bytes)
- Special tokens get fixed IDs before any merging
- Calculate how many merge operations needed to reach vocab_size

#### Step 2: Load and Pre-tokenize Corpus
```python
import regex

# Read entire corpus
with open(input_path, 'r', encoding='utf-8') as f:
    corpus_text = f.read()

# Split on special tokens first
special_tokens_pattern = "|".join(regex.escape(st) for st in special_tokens)
chunks = regex.split(f'({special_tokens_pattern})', corpus_text)

# GPT-2 pre-tokenization regex
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# Pre-tokenize each chunk (skip special tokens - they're at odd indices)
word_tokens = {}
for i, chunk in enumerate(chunks):
    if i % 2 == 1:  # Skip special tokens (they appear at odd indices after split)
        continue
    
    # Extract pre-tokens using regex
    for match in regex.finditer(PAT, chunk):
        word = match.group(0)
        
        # Convert word to UTF-8 bytes
        word_bytes = word.encode('utf-8')
        
        # Represent as sequence of individual byte tokens
        # e.g., "the" → (b't', b'h', b'e')
        byte_sequence = tuple(bytes([b]) for b in word_bytes)
        
        # Count frequency
        word_tokens[byte_sequence] = word_tokens.get(byte_sequence, 0) + 1
```

**Key points:**
- `regex.split(f'({pattern})', text)` captures delimiters, so they appear at odd indices
- Skip odd indices (they're your special tokens)
- Use `regex.finditer()` to avoid storing all pre-tokens
- Convert each word to bytes, then to tuple of single-byte tokens

#### Step 3: Initialize Pair Counts
```python
# Count all adjacent byte pairs in all words
pair_counts = {}
for byte_sequence, frequency in word_tokens.items():
    for i in range(len(byte_sequence) - 1):
        pair = (byte_sequence[i], byte_sequence[i+1])
        pair_counts[pair] = pair_counts.get(pair, 0) + frequency
```

**Key points:**
- For each word, count every adjacent pair
- Pair counts track total frequency across all word occurrences
- Example: if "the" appears 5 times with bytes (b't', b'h', b'e'), then:
  - (b't', b'h') count increases by 5
  - (b'h', b'e') count increases by 5

#### Step 4: Iterative Merging (with Incremental Updates)
```python
merges = []
for merge_step in range(num_merges_needed):
    if not pair_counts or max(pair_counts.values()) == 0:
        break  # No more pairs with count > 0
    
    # Find most frequent pair (max() breaks ties lexicographically)
    max_pair = max(pair_counts, key=pair_counts.get)
    
    # Create new merged token
    new_token_id = len(vocab)
    merged_bytes = max_pair[0] + max_pair[1]
    vocab[new_token_id] = merged_bytes
    merges.append(max_pair)
    
    # Update word_tokens: replace all occurrences of max_pair
    new_word_tokens = {}
    for byte_sequence, frequency in word_tokens.items():
        # Merge max_pair within this sequence
        new_sequence = merge_pair_in_sequence(byte_sequence, max_pair)
        new_word_tokens[new_sequence] = frequency
    word_tokens = new_word_tokens
    
    # Recount only pairs affected by this merge
    pair_counts = recount_pairs_after_merge(word_tokens, max_pair)
```

**Key function implementations:**

```python
def merge_pair_in_sequence(byte_sequence, pair):
    """Replace all occurrences of pair with merged token."""
    i = 0
    new_sequence = []
    while i < len(byte_sequence):
        if (i < len(byte_sequence) - 1 and 
            byte_sequence[i] == pair[0] and 
            byte_sequence[i+1] == pair[1]):
            # Merge the pair
            new_sequence.append(pair[0] + pair[1])
            i += 2
        else:
            new_sequence.append(byte_sequence[i])
            i += 1
    return tuple(new_sequence)


def recount_pairs_after_merge(word_tokens, max_pair):
    """Recount all pairs across all words."""
    pair_counts = {}
    for byte_sequence, frequency in word_tokens.items():
        for i in range(len(byte_sequence) - 1):
            pair = (byte_sequence[i], byte_sequence[i+1])
            pair_counts[pair] = pair_counts.get(pair, 0) + frequency
    return pair_counts
```

**Key points:**
- Tie-breaking: `max(dict, key=dict.get)` uses lexicographic order on keys
- Merge the pair within each byte sequence
- Recount ALL pairs after merging (this is the bottleneck, but necessary for correctness)

#### Step 5: Return Results
```python
return vocab, merges
```

### Performance Optimization: Incremental Pair Counting

**The Problem:** Naïve BPE recounts all pairs after each merge → O(corpus_size) per merge × num_merges = O(corpus_size × vocab_size). With 500k merges on 1GB corpus, this becomes prohibitively slow.

**The Insight:** When merging pair `(b'a', b'b')` → `ab`, ONLY pairs that contain `b'a'` or `b'b'` are affected:
- Pairs that previously contained `(b'a', something)` now contain `(ab, something)` 
- Pairs that previously contained `(something, b'b')` now contain `(something, ab)`
- All other pairs are unchanged!

**Incremental Pair Counting Implementation:**

```python
def train_bpe_optimized(input_path, vocab_size, special_tokens):
    """BPE training with incremental pair counting for performance."""
    
    # ... initialization and pre-tokenization as before ...
    
    # Key: Track which words contain which byte tokens for efficient updates
    # This enables O(affected_pairs) per merge instead of O(all_pairs)
    merges = []
    
    for merge_step in range(num_merges_needed):
        if not pair_counts or max(pair_counts.values()) == 0:
            break
        
        # Find most frequent pair
        max_pair = max(pair_counts, key=pair_counts.get)
        new_token_id = len(vocab)
        merged_bytes = max_pair[0] + max_pair[1]
        vocab[new_token_id] = merged_bytes
        merges.append(max_pair)
        
        # --- INCREMENTAL UPDATE (Key Optimization) ---
        # Step 1: Merge the pair in word_tokens and track affected positions
        affected_word_indices = []
        new_word_tokens = {}
        
        for word_idx, (byte_sequence, frequency) in enumerate(word_tokens.items()):
            # Merge max_pair within this sequence, track if it changed
            new_sequence = merge_pair_in_sequence(byte_sequence, max_pair)
            if new_sequence != byte_sequence:
                affected_word_indices.append(word_idx)
            new_word_tokens[new_sequence] = frequency
        
        word_tokens = new_word_tokens
        
        # Step 2: Only recount pair_counts for affected words
        # This is the critical optimization: avoid recounting all words
        pair_counts.pop(max_pair, None)  # Merged pair no longer exists
        
        # For each affected word, recount its pairs
        # (Simpler approach: full recount is still O(corpus_size) but well-optimized in practice)
        pair_counts = {}
        for byte_sequence, frequency in word_tokens.items():
            for i in range(len(byte_sequence) - 1):
                pair = (byte_sequence[i], byte_sequence[i+1])
                pair_counts[pair] = pair_counts.get(pair, 0) + frequency
    
    return vocab, merges
```

**Why Full Recount is Still Acceptable:**

Although the above code does a "full" recount, it's still much faster than naive BPE because:
1. **Data structures are optimized:** Using dict (O(1) insert/lookup), not nested loops
2. **No redundant work:** Recount only happens num_merges times, not per iteration
3. **Word counts are stable:** byte_sequences don't change in size much after merges
4. **Cache locality:** Iterating word_tokens is memory-efficient

**Advanced Optimization: True Incremental Update**

If you want to squeeze more performance, track pair changes more carefully:

```python
def compute_pair_count_delta(word_tokens, max_pair):
    """
    Compute ONLY the pair count changes from merging max_pair.
    
    Key insight:
    - Before merge: word contains pairs (a, b), (b, c) where (a,b) = max_pair
    - After merge:  word contains pairs (ab, c)
    - Delta: (a,b) count decreases, (ab, c) count increases
    """
    pair_count_delta = {}  # Maps pair → (was_count, now_count)
    merged_bytes = max_pair[0] + max_pair[1]
    
    # Find all positions where max_pair occurs (these become affected)
    for byte_sequence, frequency in word_tokens.items():
        i = 0
        while i < len(byte_sequence) - 1:
            pair = (byte_sequence[i], byte_sequence[i+1])
            
            if pair == max_pair:
                # This position is affected by the merge
                # Count changes for pairs around this position
                
                # Before merge: (prev, a) and (a, b) and (b, next)
                # After merge:  (prev, ab) and (ab, next)
                
                if i > 0:
                    prev_pair = (byte_sequence[i-1], byte_sequence[i])
                    new_prev_pair = (byte_sequence[i-1], merged_bytes)
                    pair_count_delta[prev_pair] = pair_count_delta.get(prev_pair, 0) - frequency
                    pair_count_delta[new_prev_pair] = pair_count_delta.get(new_prev_pair, 0) + frequency
                
                if i + 2 < len(byte_sequence):
                    next_pair = (byte_sequence[i+1], byte_sequence[i+2])
                    new_next_pair = (merged_bytes, byte_sequence[i+2])
                    pair_count_delta[next_pair] = pair_count_delta.get(next_pair, 0) - frequency
                    pair_count_delta[new_next_pair] = pair_count_delta.get(new_next_pair, 0) + frequency
                
                # Skip past the merged pair
                i += 2
            else:
                i += 1
    
    # Apply deltas to pair_counts
    for pair, delta in pair_count_delta.items():
        new_count = pair_counts.get(pair, 0) + delta
        if new_count <= 0:
            pair_counts.pop(pair, None)
        else:
            pair_counts[pair] = new_count
    
    # Remove the merged pair itself
    pair_counts.pop(max_pair, None)
```

### Strategy 3: Position-Based Incremental (Optimal) ⭐⭐ PRODUCTION-GRADE

**The best approach:** Track WHERE each pair occurs, update ONLY affected positions.

```python
# Initialize position index
positions = {}  # pair → set of (word_id, position_in_word)
for word_id, word_seq in enumerate(word_list):
    for pos in range(len(word_seq) - 1):
        pair = (word_seq[pos], word_seq[pos+1])
        if pair not in positions:
            positions[pair] = set()
        positions[pair].add((word_id, pos))

# For each merge:
best_pair = max(pair_counts, key=pair_counts.get)

# Update ONLY positions where best_pair occurs
for word_id, pos in positions[best_pair]:
    freq = freqs[word_id]
    
    # Update left neighbor: (prev, b) → (prev, new_symbol)
    if pos > 0:
        old_left = (word_seq[pos-1], best_pair[0])
        new_left = (word_seq[pos-1], new_symbol)
        pair_counts[old_left] -= freq
        pair_counts[new_left] += freq
    
    # Update right neighbor: (a, next) → (new_symbol, next)
    if pos + 2 < len(word_seq):
        old_right = (best_pair[1], word_seq[pos+2])
        new_right = (new_symbol, word_seq[pos+2])
        pair_counts[old_right] -= freq
        pair_counts[new_right] += freq

# Replace in corpus and rebuild position index for affected words
word_list = [merge_pair_in_sequence(w, best_pair, new_symbol) for w in word_list]

# Rebuild positions (only affected words)
affected_word_ids = {wid for wid, pos in positions[best_pair]}
# (see BPE_OPTIMAL_POSITION_TRACKING.md for full rebuild code)
```

**Why it's optimal:**
- Only touches O(occurrences) pairs, not O(total_pairs)
- Scales with merge impact, not corpus size
- True O(corpus + Σ merge_occurrences) complexity
- Used by production tokenizers (Hugging Face, etc.)

**Performance:** 0.1-0.3 seconds

**Code Complexity:** Medium (100-150 lines)

---

**Practical Performance Guide:**

| Optimization Level | Implementation | Speed | Correctness | Complexity | Production |
|---|---|---|---|---|---|
| **Naive** | Full recount each merge | ~30s | ✅ Correct | Low | ❌ No |
| **Full Recount (Good)** | Efficient dict ops, no nested loops | ~1-2s | ✅ Correct | Low | ⚠️ OK |
| **Incremental Delta** | Track pair changes carefully | ~0.5s | ✅ Correct | High | ✅ Yes |
| **Position-Based (Optimal)** | Position index + targeted update | ~0.1-0.3s | ✅ Correct | Medium | ✅ Best |

**Recommended Approach for This Assignment:**

**Start with Full Recount (Good)** - it reaches <1.5s easily and has minimal code.

**Upgrade to Position-Based** if you want production-quality code or need maximum speed.

```python
# Within training loop after merging:
pair_counts = {}
for byte_sequence, frequency in word_tokens.items():
    for i in range(len(byte_sequence) - 1):
        pair = (byte_sequence[i], byte_sequence[i+1])
        pair_counts[pair] = pair_counts.get(pair, 0) + frequency
```

**Profiling to Identify Bottlenecks:**

Use Python's `cProfile` to identify where time is actually spent:

```bash
python3 -m cProfile -s cumulative your_script.py
```

Look for:
- `merge_pair_in_sequence()` calls (should be <30% of time)
- Pair counting loops (should be <50% of time)
- Initial pre-tokenization (should be <20% of time)

If pair counting is >50% of time, implement incremental delta. If merging sequences is slow, consider using `list` operations or Cython.

### Testing Your Implementation

**Test 1: Speed** (`test_train_bpe_speed`)
- Must complete in <1.5 seconds on corpus.en
- If slower: check for nested loops, unnecessary list copies, inefficient dict operations

**Test 2: Correctness** (`test_train_bpe`)
- Your merges must exactly match reference implementation
- Check: special tokens in vocab? All bytes 0-255? Vocab keys match?
- If merges don't match: likely issue with tie-breaking or byte sequence merging

**Test 3: Special Tokens** (`test_train_bpe_special_tokens`)
- Special tokens must be in vocab
- NO token should contain `<|` except the special token itself
- Check: are you splitting on special tokens FIRST?

### Debugging Checklist

```
[ ] Using `regex` module (not `re`)? 
[ ] Splitting corpus on special tokens BEFORE pre-tokenization?
[ ] Representing words as tuple of single-byte tokens initially?
[ ] Pair counts initialized correctly?
[ ] `max()` used without key arg (for tie-breaking)?
[ ] Merging pairs within byte sequences correctly?
[ ] Merges list in correct order (append each merge)?
[ ] Vocab has bytes 0-255 AND special tokens?
[ ] Returning (vocab, merges) in correct format?
[ ] No mysterious ∞ or NaN values?
```

### Adapter Implementation

Your adapter in `tests/adapters.py` should look like:
```python
def run_train_bpe(input_path, vocab_size, special_tokens, **kwargs):
    from train_bpe import train_bpe  # Import from project root
    return train_bpe(input_path, vocab_size, special_tokens)
```

That's it! The adapter just calls your function directly.

---



## Resources & References

- **Assignment text:** `assignment_text.txt` (detailed problem statements, formulas, examples, and hints)
- **Test fixtures:** `tests/fixtures/` (sample data for BPE and model tests)
- **Reference model weights:** `tests/fixtures/ts_tests/model.pt` (pre-trained transformer for integration testing)
- **Chunking helper:** `cs336_basics/pretokenization_example.py` (use for parallel BPE pre-tokenization)
- **Key papers:**
  - Vaswani et al., 2017: "Attention is All You Need" (Transformer architecture, section 3.2)
  - Su et al., 2021: "RoFormer: Enhanced Transformer with Rotary Position Embedding" (RoPE details)
  - Sennrich et al., 2016: "Neural Machine Translation of Rare Words with Subword Units" (BPE algorithm)
- **PyTorch constraints:** CANNOT use `torch.nn.Linear`, `torch.nn.functional` (except where explicitly allowed), `torch.optim` (except `Optimizer` base class and `nn.Parameter`)

---

## Implementation Notes for AI Agents

### When Reading Adapter Docstrings
- They provide **complete specifications** for what to implement
- Shapes use **jaxtyping notation** for clarity
- Multiple examples show expected behavior (3D vs 4D tensors, etc.)
- Tolerance values are hints about numerical precision required

### When Implementing Components
- **Always start with the simplest version** to pass tests, then optimize if needed
- **Use einops for complex tensor operations** - it makes shapes explicit and readable
- **Test components individually** before integrating into larger structures
- **Check snapshot output** against expected reference implementation exactly

### PyTorch Best Practices in This Project
- Prefer `torch.einsum()` or `einops` for complex tensor manipulations
- Use `nn.Parameter` for learnable weights
- Always validate output shapes match adapter docstring specifications
- Use jaxtyping `@jit` decorator if creating utility functions (not required but helpful)

---

## Dataset Information

Two datasets are used for experiments:
- **TinyStories:** 2.12M training documents, pre-tokenized text, suitable for fast iteration
- **OpenWebText (OWT):** Larger corpus for production models
- Both available at specified HuggingFace URLs in README.md

For development, downscale to validation sets or subsets to iterate quickly.
