import torch
import torch.nn as nn

device=torch.device("cuda")


class Add_Norm(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.model=nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Dropout(dropout)
        )

    def forward(self, x, sublayer):
        return x + self.model(sublayer)
    


