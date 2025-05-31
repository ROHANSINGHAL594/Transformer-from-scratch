import torch
import torch.nn as nn
import math
device=torch.device("cuda")

class Positional_encoding(nn.Module):
    def __init__(self,d_model):
        super().__init__()
        
        self.d_model= d_model
    def pos_encoding(self,pos, k):
        
        f = lambda i,k: pos / 10000**(2 * (i // 2) / k)
        return [math.sin(f(i,k)) if i%2==0 else math.cos(f(i,k)) for i in range(0,k)]
    
    
    
    def forward(self,x):
        f= torch.tensor( [ self.pos_encoding(pos,self.d_model) for pos in range(x.shape[1])],device="cuda")
        return x+ f.unsqueeze(0).repeat(x.shape[0],1,1)

# tensor1= torch.tensor([[[1,2,3],[2,3,1],[3,2,4]],[[1,2,3],[2,3,1],[4,2,4]]],dtype=torch.float32).to(device)
# class1= Positional_encoding(3,3)
# print(class1.forward(tensor1))
