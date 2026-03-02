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
        """Apply the linear transformation to the input."""
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
        """ Input shape (batch_size, sequence_length, d_model)
        """
        in_dtype = x.dtype
        x = x.to(torch.float32)
        # Your code here performing RMSNorm
        rms = (x.pow(2).mean(dim=-1, keepdim=True) + self.eps).sqrt()
        result = x * self.weights / rms
        # Return the result in the original dtype
        return result.to(in_dtype)