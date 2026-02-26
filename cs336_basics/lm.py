import torch
import math
from einops import einsum

class Linear:
    def __init__(self, in_features, out_features, device=None, dtype=None):
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