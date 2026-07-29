from cs336_basics.generate import generate, main
import torch

class TestModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.call_counter = 0
    
    def forward(self, inputs, rope=None):
        res = torch.Tensor([[1, 2, 3, 4, 5, 6]])
        # Set 5th token to eot, which is voc idx 3
        if self.call_counter == 3:
            res[0][3] = 100000
        self.call_counter += 1
        return res

rope = None
model1 = TestModel()
gen_token1 = generate(model1, None, [0], 2, 10, 0.01, 0.8, 3) # should reach max
print(gen_token1)
model2 = TestModel()
gen_token2 = generate(model2, None, [0], 4, 10, 0.01, 0.8, 3) # should reach eot
print(gen_token2)

