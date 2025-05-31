import torch
import torch.nn as nn

device=torch.device("cuda")
from MultiHeadAttention import Attention
from pos_FF_Neural import PositionWiseFeedForward
from add_norm import Add_Norm


class Decoder(nn.Module):
    def __init__(self,d_model,dff,heads=8,dropout=0.1):
        super().__init__()
        self.masked_attention=Attention(d_model,heads).to(device)
        self.add_norm1= Add_Norm(d_model,dropout).to(device)
        self.add_norm2= Add_Norm(d_model,dropout).to(device)
        self.add_norm3= Add_Norm(d_model,dropout).to(device)
        self.cross_attention =Attention(d_model,heads).to(device)
        self.ffn=PositionWiseFeedForward(d_model,dff,dropout).to(device)
    
    def forward(self,y,x,mask1,mask2):
        mask_attention_output= self.masked_attention.forward(y,y,y,mask=mask2)
        add_norm_output1= self.add_norm1.forward(mask_attention_output,y)
        
        cross_attention_output=self.cross_attention.forward(add_norm_output1,x,x,mask=mask1)
        add_norm_output2=self.add_norm2.forward(cross_attention_output,add_norm_output1)
        ffn_output= self.ffn.forward(add_norm_output2)
        add_norm_output3= self.add_norm3.forward(ffn_output,add_norm_output2)
        return add_norm_output3

        
