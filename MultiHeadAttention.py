import torch
import torch.nn as nn
from math import sqrt
device=torch.device("cuda")
class Attention(nn.Module):
    def __init__(self,d_model,heads):
        super().__init__()
        self.h=heads
        self.d_k=d_model//heads
        self.d_model= d_model
        
        self.query_weight = nn.Linear(d_model, d_model,bias=False).to(device)
        self.key_weight = nn.Linear(d_model,d_model,bias=False).to(device)
        self.value_weight = nn.Linear(d_model, d_model,bias=False).to(device)

        self.unify =nn.Linear(d_model, d_model).to(device)
    
        

    def forward(self,query,key,value,mask=None):
        
        batch_size, sequence_length, d_model= query.shape
        _, key_len, _ = key.shape
        query_vector=self.query_weight(query)
        key_vector= self.key_weight(key)
        value_vector=self.value_weight(value)

        query_vector= query_vector.view(batch_size,sequence_length,self.h,self.d_k).transpose(1,2)
        key_vector=key_vector.view(batch_size,key_len,self.h,self.d_k).transpose(1,2)
        value_vector=value_vector.view(batch_size,key_len,self.h,self.d_k).transpose(1,2)
        query_key_mul= torch.matmul(query_vector,key_vector.transpose(-1,-2))/sqrt(self.d_k)
        if mask is not None:
            query_key_mul= query_key_mul.masked_fill(mask==0,-1e9)
            
        query_key_mul=query_key_mul.softmax(dim=-1)
        self_attention_output= torch.matmul(query_key_mul,value_vector).transpose(1,2).contiguous().view(batch_size,sequence_length,self.h*self.d_k)
        
        
        return self.unify(self_attention_output)
    

if __name__ == "__main__":
    tensor1= torch.tensor([[[1,2,3],[2,3,1],[3,2,4]],[[1,2,3],[2,3,1],[4,2,4]]],dtype=torch.float32).to(device)
    class1= Attention(3,1)
    print(class1.forward(tensor1,tensor1,tensor1))

