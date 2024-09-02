# Import the relevant package
import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

device = 'cuda'
# or you can set your device to 'cpu' if you don'have any GPU

# train epochs
epochs = 100

# train dataset
sentences = [
    # 中文和英语的单词个数不要求相同
    # enc_input                dec_input           dec_output
    ['我 有 一 个 好 朋 友 P', 'S i have a good friend .', 'i have a good friend . E'],
    ['我 有 零 个 女 朋 友 P', 'S i have zero girl friend .', 'i have zero girl friend . E']
]

# test dataset
# input : ['我 有 一 个 女 朋 友 P']
# output : ['i have a girl friend . E']

# Build the vocab
# source vocab
src_vocab = {'P': 0, '我': 1, '有': 2, '一': 3, '个': 4, '好': 5, '朋': 6, '友': 7, '零': 8, '女': 9}
src_idx2word = {i: w for i, w in enumerate(src_vocab)}
src_vocab_size = len(src_vocab)
# taget vocab
tgt_vocab = {'P': 0, 'i': 1, 'have': 2, 'a': 3, 'good': 4, 'friend': 5, 'zero': 6, 'girl': 7, '.': 8, 'E': 9, 'S': 10}
idx2word = {i: w for i, w in enumerate(tgt_vocab)}
tgt_vocab_size = len(tgt_vocab)

src_len = 8 # enc_input max sequence length
tgt_len = 7 # dec_input max sequence length

# Transformer Parameters
d_model = 512 # Embedding Size
d_ff = 2048 # FeedForward dimension
d_k = d_v = 64 # dimension of K(=Q), V
n_layers = 6 # number of Encoder of Decoder Layers
n_heads = 8 # number of heads in Multi-Head Attention

# create the dataset
def creat_dataset(senences):
    """Create the dataset: convert the sentence into index."""
    enc_input, dec_input, dec_output = [], [], []
    # Go through each sentence
    for i in range(len(senences)):
        enc_input = [[src_vocab[n] for n in senences[i][0].split()]]
        dec_input = [[tgt_vocab[n] for n in senences[i][1].split()]]
        dec_output = [[tgt_vocab[n] for n in senences[i][2].split()]]
    # convert the list into tensor
    return torch.LongTensor(enc_input), torch.LongTensor(dec_input), torch.LongTensor(dec_output)

enc_inputs, dec_inputs, dec_outputs = creat_dataset(sentences)

class MyDataSet(Data.Dataset):
    """self-define dataset"""
    def __init__(self, enc_inputs, dec_inputs, dec_outputs):
        super(MyDataSet, self).__init__()
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs

    def __getitem__(self, index):
        return self.enc_inputs[index], self.dec_inputs[index], self.dec_outputs[index]

    def __len__(self):
        return len(self.enc_inputs)

loader = Data.DataLoader(dataset=MyDataSet(enc_inputs, dec_inputs, dec_outputs), batch_size=2, shuffle=True)

# Define Transformer model

class PositionalEncoding(nn.Module):
    """
    Implement the PE function
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        :param d_model: the dimension of the embedding
        :param dropout: the dropout value
        :param max_len: the maximum length of the input sequence

        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # max_len is the max_len of a input sequence, d_model is the dimension of the embedding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # if the column index is even, then use sin, else use cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    def forward(self, x):
        """
        :param x: the input sequence: [batch_size, seq_len, d_model]
        """
        # using residual connection
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


# Define the attention padding mask
def get_attn_pad_mask(seq_q, seq_k):
    """
    :param seq_q: [batch_size, seq_len]
    :param seq_k: [batch_size, seq_len]
    :return: [batch_size, len_q, len_k]
    """
    batch_size, len_q = seq_q.size()    # the seq.q is for expand dim
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
    return pad_attn_mask.expand(batch_size, len_q, len_k)

def get_attn_subsequent_mask(seq):
    """
    :param seq: [batch_size, tgt_len]
    :return: [batch_size, tgt_len, tgt_len]
    """
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    # attn_shape: (batch_size, tgt_len, tgt_len)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    subsequent_mask = torch.from_numpy(subsequent_mask)
    return subsequent_mask

# Define the scale dot product attention
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        """
        :param Q: [batch_size, n_heads, len_q, d_k]
        :param K: [batch_size, n_heads, len_k, d_k]
        :param V: [batch_size, n_heads, len_v(=len_k), d_v]
        :param attn_mask: [batch_size, n_heads, seq_len, seq_len]

        """
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(d_k) # scores : [batch_size, n_heads, len_q, len_k]
        # using mask matrix to padding scores
        scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is True.
        attn = nn.Softmax(dim=-1)(scores) # scores : [batch_size, n_heads, len_q, len_k]
        context = torch.matmul(attn, V) # [batch_size, n_heads, len_q, d_v]
        return context, attn

class MultiHeadAttention(nn.Module):
    """Multi-head attention module"""
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        # keep the output shape is (batch_size, seq_len, d_model)
        self.fc = nn.Linear(d_v * n_heads, d_model, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        """
        :param input_Q: [batch_size, len_q, d_model]
        :param input_K: [batch_size, len_k, d_model]
        :param input_V: [batch_size, len_v(=len_k), d_model]
        """
        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D') -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # (B, H, S, W)
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)

        # expand the mask mritrix to fit the shape of Q
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) # (B, H, L_q, L_k)
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v)
        output = self.fc(context)  # [batch_size, len_q, d_model]
        return nn.LayerNorm(d_model).to(device)(output + residual)

# Define a class for the position-wise feed-forward network
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )

    def forward(self, inputs):
        """
        :param inputs: [batch_size, seq_len, d_model]
        """
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(d_model).to(device)(output + residual)

# Define the encoder layer
class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        """
        :param enc_inputs: [batch_size, src_len, d_model]
        :param enc_self_attn_mask: [batch_size, src_len, src_len]
        """
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs, attn

# Define the decoder layer
class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, dec_input, enc_output, dec_self_attn_mask, dec_enc_attn_mask):
        """
        :param dec_input: [batch_size, tgt_len, d_model]
        :param enc_output: [batch_size, src_len, d_model]
        :param dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
        :param dec_enc_attn_mask: [batch_size, tgt_len, src_len]
        """
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_input, dec_input, dec_input, dec_self_attn_mask)
        # For the encoder-decoder attention, the query is from the decoder, but the key and value are from the encoder.
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_output, enc_output, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn, dec_enc_attn

# Define the Encoder
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(src_vocab_size, d_model)    # token embedding
        self.pos_emb = PositionalEncoding(d_model)            # position embedding
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)]) # stacking n layers

    def forward(self, enc_inputs):
        """
        :param enc_inputs: [batch_size, src_len]
        """
        enc_outputs = self.src_emb(enc_inputs)
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1).to(device)
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs).to(device)
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)

            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns

# Define the Decoder
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)    # token embedding
        self.pos_emb = PositionalEncoding(d_model)            # position embedding
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)]) # stacking n layers

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        """
        :param dec_inputs: [batch_size, tgt_len]
        :param enc_outputs: [batch_size, src_len, d_model]
        :param dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
        :param dec_enc_attn_mask: [batch_size, tgt_len, src_len]
        """
        dec_outputs = self.tgt_emb(dec_inputs)
        dec_outputs = self.pos_emb(dec_outputs.transpose(0, 1)).transpose(0, 1).to(device)
        dec_self_attns_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs).to(device)
        dec_self_attns_subsequent_mask = get_attn_subsequent_mask(dec_inputs).to(device)
        dec_enc_attn_mask = torch.gt((dec_self_attns_pad_mask + dec_self_attns_subsequent_mask), 0).to(device)
        dec_self_attn_mask = get_attn_pad_mask(dec_inputs, enc_outputs)

        dec_self_attns, dec_enc_attens = [], []
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attens.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attens

# Define the Transformer
class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False)

    def forward(self, enc_inputs, dec_inputs):
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_outputs)
        dec_logits = self.projection(dec_outputs)
        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns


learning_rate = 1e-3
momentum = 0.99

model = Transformer().to(device)
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

for epoch in range(epochs):
    for enc_inputs, dec_inputs, dec_outputs in loader:
        """
        enc_inputs: [batch_size, src_len]
        dec_inputs: [batch_size, tgt_len]
        dec_outputs: [batch_size, tgt_len]"""
        enc_inputs, dec_inputs, dec_outputs = enc_inputs.to(device), dec_inputs.to(device), dec_outputs.to(device)
        outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
        loss = criterion(outputs, dec_outputs.view(-1))
        print("Epoch:", "%04d"% (epoch + 1), "loss = ", "{:.6f}".format(loss))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def greedy_decoder(model, enc_inputs, start_symbol):
    """greedy encoder"""
    enc_outputs, enc_self_attns = model.encoder(enc_inputs)
    dec_inputs = torch.zeros(1, 0).type_as(enc_inputs.data)
    terminal = False
    next_symbol = start_symbol
    while not terminal:
        dec_inputs = torch.cat([dec_inputs.to(device), torch.tensor([[next_symbol]], dtype=enc_inputs.dtype).to(device)], -1)
        dec_outputs, _, _ = model.decoder(dec_inputs, enc_outputs)
        projected = model.projection(dec_outputs)
        prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]

        next_word = prob.data[-1]
        next_symbol = next_word
        if next_symbol == 1:
            terminal = True

    greed_dec_predict = dec_inputs[0, 1:]
    return greed_dec_predict

sentences = [
    # enc_input                dec_input           dec_output
    ['我 有 零 个 女 朋 友 P', '', '']
]

enc_inputs, dec_inputs, dec_outputs = creat_dataset(sentences)
test_loader = Data.DataLoader(MyDataSet(enc_inputs, dec_inputs, dec_outputs), 2, True)
enc_inputs, _, _ = next(iter(test_loader))

print()
print("="*30)
print("利用训练好的Transformer模型将中文句子'我 有 零 个 女 朋 友' 翻译成英文句子: ")
for i in range(len(enc_inputs)):
    greedy_dec_predict = greedy_decoder(model, enc_inputs[i].view(1, -1).to(device), start_symbol=tgt_vocab["S"])
    print(enc_inputs[i], '->', greedy_dec_predict.squeeze())
    print([src_idx2word[t.item()] for t in enc_inputs[i]], '->',
          [idx2word[n.item()] for n in greedy_dec_predict.squeeze()])


