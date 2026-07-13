from lm import softmax
import torch
from typing import List

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
    
def generate(model, prompt, max_new_tokens, temp, top_p_threshold, eot_token) -> List[int]:
    generated_tokens = []
    while len(generated_tokens) < max_new_tokens:
        inputs = torch.concat([torch.tensor(prompt), torch.tensor(generated_tokens)])
        logits = model(inputs) # [token, vocab]
        last_token_repr = logits[-1] # [vocab]
        dist = temp_softmax(last_token_repr, temp, dim=-1) # temperatured softmax
        top_p_sampling(dist, top_p_threshold)
        token = torch.multinomial(dist, 1)[0] 
        if token == eot_token:
            return generated_tokens
        generated_tokens.append(token)
    return generated_tokens

# toy test
class TestModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.call_counter = 0
    
    def forward(self, inputs):
        res = torch.Tensor([[1, 2, 3, 4, 5, 6]])
        # Set 5th token to eot, which is voc idx 3
        if self.call_counter == 3:
            res[0][3] = 100000
        self.call_counter += 1
        return res

model1 = TestModel()
gen_token1 = generate(model1, [0], 2, 0.01, 0.8, 3) # should reach max
print(gen_token1)
model2 = TestModel()
gen_token2 = generate(model2, [0], 4, 0.01, 0.8, 3) # should reach eot
print(gen_token2)