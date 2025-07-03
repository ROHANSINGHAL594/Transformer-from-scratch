import torch
import torch.nn as nn

device=torch.device("cuda")


from positional_encoding import Positional_encoding
from Encoder import Encoder
from decoder import Decoder

class Transformer(nn.Module):
    def __init__(self,d_model,input_vocab_size,target_vocab_size,dff=2048,heads=8,number_layers=6,dropout=0.1):
        super().__init__()
        self.encoder_embedding= nn.Embedding(num_embeddings=input_vocab_size,embedding_dim=d_model,padding_idx=0).to(device)
        self.decoder_embedding= nn.Embedding(num_embeddings=target_vocab_size,embedding_dim=d_model,padding_idx=0).to(device)
        self.positional_encoding=Positional_encoding(d_model).to(device)
        self.encoder_layers= nn.ModuleList([Encoder(d_model,dff,heads,dropout) for _ in range(number_layers)]).to(device)
        self.decoder_layers=nn.ModuleList([Decoder(d_model,dff,heads,dropout) for _ in range(number_layers)]).to(device)
        self.fc = nn.Linear(d_model,target_vocab_size).to(device)
        self.dropout = nn.Dropout(dropout).to(device)


    def generate_mask(self, src, tgt):
        # src and tgt are (seq_len, batch_size)
        src_seq_len = src.size(0)
        tgt_seq_len = tgt.size(0)

        # src_mask: (batch_size, 1, 1, src_seq_len)
        src_mask = (src != 0).transpose(0, 1).unsqueeze(1).unsqueeze(2).to(device)

        # tgt_mask: (batch_size, 1, tgt_seq_len, tgt_seq_len)
        tgt_padding_mask = (tgt != 0).transpose(0, 1).unsqueeze(1).unsqueeze(2).to(device)
        
        nopeak_mask = (1 - torch.triu(torch.ones(1, tgt_seq_len, tgt_seq_len), diagonal=1)).bool().to(device)
        tgt_mask = tgt_padding_mask & nopeak_mask
        return src_mask.to(device), tgt_mask.to(device)
        

    def forward(self,x,y):
        x_mask,y_mask=self.generate_mask(x,y)
        
        x_embedding=self.encoder_embedding(x)
        y_embedding=self.decoder_embedding(y)

        x_positional_encoding=self.dropout(self.positional_encoding.forward(x_embedding))
        y_positional_encoding= self.dropout(self.positional_encoding.forward(y_embedding))
        
        encoder_output= x_positional_encoding
        for layer in self.encoder_layers:
            encoder_output=layer.forward(encoder_output,x_mask)
        
        decoder_output= y_positional_encoding
        for layer in self.decoder_layers:
            decoder_output=layer.forward(decoder_output,encoder_output,x_mask,y_mask)

        linear_layer_output= self.fc(decoder_output)
        return linear_layer_output
    
