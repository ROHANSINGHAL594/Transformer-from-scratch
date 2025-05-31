import torch
import torch.nn as nn

device=torch.device("cuda")
from MultiHeadAttention import Attention
from pos_FF_Neural import PositionWiseFeedForward
from add_norm import Add_Norm

class Encoder(nn.Module):
    def __init__(self,d_model,dff,heads=8 ,dropout=0.1):
        super().__init__()
        self.attention=Attention(d_model,heads).to(device)
        self.add_norm1= Add_Norm(d_model,dropout).to(device)
        self.ffn = PositionWiseFeedForward(d_model,dff,dropout).to(device)
        self.add_norm2=Add_Norm(d_model,dropout).to(device)
    
    def forward(self,x,input_mask):
        attention_output = self.attention.forward(x,x,x,input_mask)
        add_norm_output1= self.add_norm1.forward(attention_output,x)
        ffn_output= self.ffn.forward(add_norm_output1)
        add_norm_output2=self.add_norm2.forward(ffn_output,add_norm_output1)
        return add_norm_output2

