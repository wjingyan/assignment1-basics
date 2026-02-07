# CS336 Assignment 1: Transformer LM From-Scratch Rules

You are helping me complete a "from-scratch" implementation of a Transformer. Accuracy to the assignment constraints is more important than using high-level libraries.

## 1. Strict Architectural Constraints
- [cite_start]**Prohibited Libraries:** You MUST NOT use any definitions from `torch.nn`, `torch.nn.functional`, or `torch.optim`[cite: 20].
- **Exceptions (The ONLY allowed items):**
    - [cite_start]`torch.nn.Parameter` [cite: 21]
    - [cite_start]Container classes: `torch.nn.Module`, `torch.nn.ModuleList`, `torch.nn.Sequential` [cite: 22, 26]
    - [cite_start]The `torch.optim.Optimizer` base class [cite: 23]
- [cite_start]**Initialization:** Use `torch.nn.init.trunc_normal_` for weight initialization[cite: 518].

## 2. Implementation Style & Best Practices
- [cite_start]**Einsum Notation:** Prioritize `einops.einsum` and `einops.rearrange` for all tensor operations to ensure readable and robust code[cite: 401, 406].
- [cite_start]**Type Hinting:** Use clear type hints for all function signatures, including tensor shapes if possible (e.g., using `jaxtyping`)[cite: 495].
- [cite_start]**Memory Efficiency:** For the Tokenizer and Data Loader, prioritize lazy generation and `np.memmap` to handle large datasets without fitting them in memory[cite: 314, 1006].

## 3. Workflow & Logic Protection
- **One Problem at a Time:** I will implement the sub-assignments sequentially (§2 Tokenizer, §3 Architecture, §4 Training, §5 Loop).
- **Code Integrity:** NEVER modify code in a file that is marked as "Completed" or "Validated." If you need to reference logic from a previous sub-problem (e.g., using the Tokenizer in the Training Loop), refer to the existing file using its path rather than rewriting it.
- **No Documentation Bloat:** Do not generate extra `.md` files, progress reports, or extensive docstrings unless explicitly asked. [cite_start]Focus entirely on providing functional Python code that passes the provided `test_*.py` files[cite: 37].

## 4. Specific Component Rules
- **Tokenizer:** Use the GPT-2 regex pattern `PAT` provided in the handout[cite: 155].
- **RMSNorm:** Always upcast inputs to `torch.float32` before squaring to prevent overflow[cite: 589].
- **Attention:** Implement "Causal" masking to prevent the model from attending to future tokens[cite: 734].
- **Linear Layers:** Implement a custom `Linear` class without a bias term[cite: 523, 529].