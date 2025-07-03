
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import spacy
import pandas as pd
from collections import Counter


try:
    spacy_fr = spacy.load('fr_core_news_sm')
except OSError:
    print('Downloading language model for the spaCy POS tagger\n'
        "(don't worry, this will only happen once)")
    from spacy.cli import download
    download('fr_core_news_sm')
    spacy_fr = spacy.load('fr_core_news_sm')

spacy_en = spacy.load('en_core_web_sm')


def tokenize_fr(text):
    return [tok.text for tok in spacy_fr.tokenizer(text)]

def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]


class Vocab:
    def __init__(self, freq_threshold, max_size):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {v: k for k, v in self.itos.items()}
        self.freq_threshold = freq_threshold
        self.max_size = max_size

    def __len__(self):
        return len(self.itos)

    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        idx = 4

        for sentence in sentence_list:
            for word in tokenize_en(sentence):
                frequencies[word] += 1

        frequencies = {k: v for k, v in frequencies.items() if v > self.freq_threshold}
        frequencies = dict(sorted(frequencies.items(), key=lambda x: -x[1])[:self.max_size-idx])
        
        for word in frequencies:
            self.stoi[word] = idx
            self.itos[idx] = word
            idx += 1

    def numericalize(self, text):
        tokenized_text = tokenize_en(text)
        return [self.stoi[token] if token in self.stoi else self.stoi["<UNK>"] for token in tokenized_text]

# --- Data Loading and Preprocessing ---

class EngFrDataset(Dataset):
    def __init__(self, csv_file, src_vocab, tgt_vocab):
        self.df = pd.read_csv(csv_file)
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        src_sentence = self.df.iloc[idx]['French words/sentences']
        tgt_sentence = self.df.iloc[idx]['English words/sentences']
        
        numericalized_src = [self.src_vocab.stoi["<SOS>"]] + self.src_vocab.numericalize(src_sentence) + [self.src_vocab.stoi["<EOS>"]]
        numericalized_tgt = [self.tgt_vocab.stoi["<SOS>"]] + self.tgt_vocab.numericalize(tgt_sentence) + [self.tgt_vocab.stoi["<EOS>"]]
        
        return torch.tensor(numericalized_src), torch.tensor(numericalized_tgt)

# Collate function to process a batch of data
class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        src = [item[0] for item in batch]
        tgt = [item[1] for item in batch]
        src = nn.utils.rnn.pad_sequence(src, batch_first=False, padding_value=self.pad_idx)
        tgt = nn.utils.rnn.pad_sequence(tgt, batch_first=False, padding_value=self.pad_idx)
        return src, tgt

def train_model():
    # Model parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv('C:/transformerproject/Attention is all you need/eng-french.csv')
    
    # Build vocab
    freq_threshold = 2
    max_vocab_size = 10000
    src_vocab = Vocab(freq_threshold, max_vocab_size)
    tgt_vocab = Vocab(freq_threshold, max_vocab_size)
    src_vocab.build_vocabulary(df['French words/sentences'].tolist())
    tgt_vocab.build_vocabulary(df['English words/sentences'].tolist())

    input_vocab_size = len(src_vocab)
    tgt_vocab_size = len(tgt_vocab)
    d_model = 512
    num_heads = 8
    num_layers = 3 # Reduced for faster training
    d_ff = 512 # Reduced for faster training
    dropout = 0.1
    batch_size = 128

    print(f"Using device: {device}")

    # Instantiate the model
    from transformer import Transformer
    transformer = Transformer(
        d_model=d_model,
        input_vocab_size=input_vocab_size,
        target_vocab_size=tgt_vocab_size,
        dff=d_ff,
        heads=num_heads,
        number_layers=num_layers,
        dropout=dropout
    ).to(device)

    # Data loader
    train_dataset = EngFrDataset('C:/transformerproject/Attention is all you need/eng-french.csv', src_vocab, tgt_vocab)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=MyCollate(pad_idx=src_vocab.stoi["<PAD>"]))

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=src_vocab.stoi["<PAD>"])
    optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    # Training loop
    transformer.train()
    print("Starting training...")
    for epoch in range(10):
        losses = 0
        for src, tgt in train_dataloader:
            src = src.to(device)
            tgt = tgt.to(device)

            tgt_input = tgt[:-1, :]

            optimizer.zero_grad()

            logits = transformer(src, tgt_input)

            tgt_out = tgt[1:, :]
            loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            loss.backward()

            optimizer.step()
            losses += loss.item()

        print(f"Epoch: {epoch+1:02d}, Loss: {losses / len(list(train_dataloader)):.4f}")

    print("-" * 50)
    print("Training finished.")

if __name__ == "__main__":
    train_model()
