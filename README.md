# 📢📢ATTENTION  IS ALL YOU NEED IMPLEMENTATION FROM SCRATCH USING PYTORCH 🧠🧠 

This Repository aims to implement the  research paper [Attention is all you need 2017](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) using pytorch .

## A little introduction about Transformers 🤖🤖
_Transformers were introduced in the 2017 paper "Attention is All You Need" by Vaswani , **revolutionizing** natural language processing. Unlike previous RNN-based models, transformers use self-attention mechanisms to process entire sequences in parallel, enabling faster and more effective learning of long-range dependencies. Initially developed for machine translation, transformers quickly became foundational in models like BERT, GPT, and T5, powering breakthroughs in language understanding, generation, and even in fields beyond NLP, such as vision and bioinformatics. Their modular architecture, scalability, and efficiency in handling large datasets have made transformers the dominant deep learning architecture in AI today._

## 📁 Features
- **Multi-head attention**
- **Add-norm layer**
- **Positional encoding**
- **Position feed forward neural network**
- **Decoder block**
- **Encoder block**
- **Transformer (arranging all the things together)**

<br>

## 🛠️ Installation
    
    git clone https://github.com/ROHANSINGHAL594/Transformer-from-scratch.git
    cd Transformer-from-scratch
    pip install -r requirements.txt

### USE THE MODEL
    from transformer import Transformer

    test_tranformer = Transformer(d_model=512,input_vocab_size=5000,target_vocab_size=5000,dff=2048,heads=8,number_layers=6,dropout=0.1)

## 📊  TRAINING
#### I made the model  and trained it on a random data ,i had to take small data as i lack a good gpu .

#### The model was working perfectly. you can use the model for your data training 
    # input should be a padded sequence of sentences  the model does the embedding and positional encoding by itself for example
    input = torch.tensor([[1,2,3,0,0].[1,4,5,6,0],.....],device=device)
    # same format for the output 

    # now the forward implementation should include
    output = transformer.forward(input,output)

    ##the rest test implementation you can check in "tranformer.py" file

## Refrences
- [Attention is all you need 2017](https://proceedings.neurips.cc/paper_files/paper/2017/file/)
- Pytorch Documentation

##  🚧Future Improvements
 - Add BLEU score evaluation

 - training for real translation datasets

 - Save/load checkpoints

 - Visualize attention weights

## 🙋‍♂️ Author's Note
This is my first advanced paper implementation where I implemented the Transformer architecture from scratch. The goal was to deeply understand how each component—like multi-head attention and positional encoding—works under the hood. i really learnt a lot from it , had many bugs in code but in the end it all worked out fine.

I'm open to feedback and collaboration, and I hope this project helps others who are learning Transformers too!




    
    




