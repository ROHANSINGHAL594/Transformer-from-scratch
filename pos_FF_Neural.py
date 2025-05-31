import torch
import torch.nn as nn

device=torch.device("cuda")

class PositionWiseFeedForward(nn.Module):
    def __init__(self,d_model,dff,dropout=0.1):
        super().__init__()
        self.model=nn.Sequential(
            nn.Linear(d_model,dff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dff,d_model)

        )

    def forward(self,x):
        return self.model(x)