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
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2).to(device)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3).to(device)
        seq_length = tgt.size(1)
        
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool().to(device)
        tgt_mask = tgt_mask & nopeak_mask
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
    
if __name__ == "__main__":
    import torch.optim as optim


    input_vocab_size = 512
    tgt_vocab_size = 512
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    max_seq_length = 100
    dropout = 0.1

    transformer = Transformer(d_model, input_vocab_size, tgt_vocab_size).to(device)

    # Generate random sample data
    src_data = torch.randint(1, input_vocab_size, (64, max_seq_length)).to(device)
    tgt_data = torch.randint(1, tgt_vocab_size, (64, max_seq_length)).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    transformer.train()

    for epoch in range(20):
        optimizer.zero_grad()
        
        tgt_input = tgt_data[:, :-1]
        tgt_output = tgt_data[:, 1:]

        output = transformer(src_data, tgt_input)  # [batch, seq_len-1, vocab_size]
        
        loss = criterion(output.contiguous().view(-1, tgt_vocab_size),
                        tgt_output.contiguous().view(-1))

        loss.backward()
        optimizer.step()

        print(f"Epoch: {epoch+1}, Loss: {loss.item():.4f}")


    print('-'*50)
    transformer.eval()

   
    val_src_data = torch.randint(1, input_vocab_size, (64, max_seq_length)).to(device)  
    val_tgt_data = torch.randint(1, tgt_vocab_size, (64, max_seq_length)).to(device) 

    with torch.no_grad():

        val_output = transformer(val_src_data, val_tgt_data[:, :-1])
        val_loss = criterion(val_output.contiguous().view(-1, tgt_vocab_size), val_tgt_data[:, 1:].contiguous().view(-1))
        print(f"Validation Loss: {val_loss.item()}")