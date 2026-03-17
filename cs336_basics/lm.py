import torch
import math
from einops import einsum

class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        
        # Initialize weight tensor
        self.weight = torch.nn.Parameter(
            torch.empty((in_features, out_features), device=device, dtype=dtype)
        )
        
        # Apply Xavier Truncated Normal initialization
        sigma = math.sqrt(2.0 / (in_features + out_features))
        torch.nn.init.trunc_normal_(
            self.weight, mean=0.0, std=sigma, a=-3.0 * sigma, b=3.0 * sigma
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(..., in_features) -> (..., out_features)"""
        return einsum(x, self.weight, "... d_in, d_in d_out -> ... d_out")

class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype

        # Initialize embedding tensor
        self.embedding = torch.nn.Parameter(
            torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        )

        # Apply Xavier Truncated Normal initialization
        sigma = 1
        torch.nn.init.trunc_normal_(
            self.embedding, mean=0.0, std=1, a=-3.0, b=3.0
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """(batch_size, seq_len) -> (batch_size, seq_len, embedding_dim)"""
        return self.embedding[token_ids]

class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype

        self.weights = torch.nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)"""
        in_dtype = x.dtype
        x = x.to(torch.float32)
        # Your code here performing RMSNorm
        rms = (x.pow(2).mean(dim=-1, keepdim=True) + self.eps).sqrt()
        result = x * self.weights / rms
        # Return the result in the original dtype
        return result.to(in_dtype)

class FeedForwardNetwork(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int,device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.device = device
        self.dtype = dtype

        # Initialize weight tensor
        self.w1 = torch.nn.Parameter(
            torch.empty((d_model, d_ff), device=device, dtype=dtype)
        )
        self.w2 = torch.nn.Parameter(
            torch.empty((d_ff, d_model), device=device, dtype=dtype)
        )
        self.w3 = torch.nn.Parameter(
            torch.empty((d_model, d_ff), device=device, dtype=dtype)
        )

        # Apply Xavier Truncated Normal initialization
        sigma = math.sqrt(2.0 / (d_model + d_ff))
        torch.nn.init.trunc_normal_(
            self.w1, mean=0.0, std=sigma, a=-3.0 * sigma, b=3.0 * sigma
        )
        torch.nn.init.trunc_normal_(
            self.w2, mean=0.0, std=sigma, a=-3.0 * sigma, b=3.0 * sigma
        )
        torch.nn.init.trunc_normal_(
            self.w2, mean=0.0, std=sigma, a=-3.0 * sigma, b=3.0 * sigma
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)"""
        w1_x = einsum(x, self.w1, "... d_model, d_model d_ff -> ... d_ff")
        w3_x = einsum(x, self.w3, "... d_model, d_model d_ff -> ... d_ff")
        silu = w1_x * torch.sigmoid(w1_x)
        elementwise_mul = silu * w3_x
        return einsum(elementwise_mul, self.w2, "... d_ff, d_ff d_model -> ... d_model")

class RoPE(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        # Create theta_i
        dims = torch.arange(0, d_k, 2).float().to(device)
        freqs = 1 / (theta ** (dims / d_k))

        # Precompute angle
        t = torch.arange(max_seq_len, device=device).float()
        # Q can be replaced with einsum?
        angles = torch.outer(t, freqs)

        self.register_buffer("cos", torch.cos(angles))
        self.register_buffer("sin", torch.sin(angles))

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """(..., seq_len, d_k) -> (..., seq_len, d_k)"""
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]

        # 2. Repeat cos and sin so they match the d_k dimension
        # (..., d_k // 2) -> (..., d_k) by repeating each value twice
        # This makes cos look like [c0, c0, c1, c1, ...]
        cos = torch.repeat_interleave(cos, 2, dim=-1)
        sin = torch.repeat_interleave(sin, 2, dim=-1)

        # 3. Create the "Interleaved" negative counterpart
        # We want x_interleaved = [-x1, x0, -x3, x2, ...]
        x_neg = torch.empty_like(x)
        x_neg[..., 0::2] = -x[..., 1::2]
        x_neg[..., 1::2] = x[..., 0::2]

        # 4. Apply the rotation: x * cos + x_neg * sin
        # This perfectly implements:
        # out_even = x_even * cos - x_odd * sin
        # out_odd  = x_even * sin + x_odd * cos
        return x * cos + x_neg * sin

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """ Apply softmax to x[dim]
    (...) -> (...)"""
    max_x = x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x - max_x)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Q: (..., seq_len_q, d_k), K: (..., seq_len_k, d_k), V: (..., seq_len_k, d_v) -> (..., seq_len_q, d_v)"""
    d_k = Q.shape[-1]
    scores = einsum(Q, K, "... seq_len_q d_k, ... seq_len_k d_k -> ... seq_len_q seq_len_k") / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    attn = softmax(scores, dim=-1)
    return einsum(attn, V, "... seq_len_q seq_len_k, ... seq_len_k d_v -> ... seq_len_q d_v")

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.device = device
        self.dtype = dtype

        # Combined projections for all heads: (d_model, d_model)
        self.q_proj = torch.nn.Parameter(
            torch.empty((d_model, d_model), device=device, dtype=dtype)
        )
        self.k_proj = torch.nn.Parameter(
            torch.empty((d_model, d_model), device=device, dtype=dtype)
        )
        self.v_proj = torch.nn.Parameter(
            torch.empty((d_model, d_model), device=device, dtype=dtype)
        )
        self.o_proj = torch.nn.Parameter(
            torch.empty((d_model, d_model), device=device, dtype=dtype)
        )

    def forward(self, x: torch.Tensor, rope: RoPE | None = None, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        """(batch, seq_len, d_model) -> (batch, seq_len, d_model)"""
        batch, seq_len, _ = x.shape
        d_k = self.d_k

        # Project to queries, keys, values
        q = einsum(x, self.q_proj, "... d_model, d_model d_out -> ... d_out")
        k = einsum(x, self.k_proj, "... d_model, d_model d_out -> ... d_out")
        v = einsum(x, self.v_proj, "... d_model, d_model d_out -> ... d_out")

        # Reshape to (batch, num_heads, seq_len, d_k)
        q = q.view(batch, seq_len, self.num_heads, d_k).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_heads, d_k).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_heads, d_k).transpose(1, 2)

        # Apply RoPE after splitting heads
        if rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)
            q = rope(q, token_positions)
            k = rope(k, token_positions)

        # Causal mask
        mask = ~torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1)

        # Attention: (batch, num_heads, seq_len, d_k)
        attn = scaled_dot_product_attention(q, k, v, mask)

        # Reshape back to (batch, seq_len, d_model)
        attn = attn.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)

        return einsum(attn, self.o_proj, "... d_model, d_model d_out -> ... d_out")

class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.device = device
        self.dtype = dtype
        self.attn = MultiHeadAttention(d_model, num_heads, device=device, dtype=dtype)
        self.ffn = FeedForwardNetwork(d_model, d_ff, device=device, dtype=dtype)
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, rope: RoPE | None = None, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        """(batch, seq_len, d_model) -> (batch, seq_len, d_model)"""
        sublayer1 = self.ln1(x)
        sublayer1 = x + self.attn(sublayer1, rope)
        sublayer2 = self.ln2(sublayer1)
        sublayer2 = sublayer1 + self.ffn(sublayer2)
        return sublayer2

class TransformerLM(torch.nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.device = device
        self.dtype = dtype
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        for i in range(num_layers):
            self.add_module(f"transformer_block_{i}", TransformerBlock(d_model, num_heads, d_ff, device=device, dtype=dtype))
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)
    
    def forward(self, x: torch.Tensor, rope: RoPE | None = None, token_positions: torch.Tensor | None = None):
        """(batch, seq_len) -> (batch, seq_len, vocab_size)"""
        emb = self.token_embeddings(x)
        for i in range(self.num_layers):
            emb = getattr(self, f"transformer_block_{i}")(emb, rope)
        normed_x = self.ln_final(emb)
        return self.lm_head(normed_x)