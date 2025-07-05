import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, dropout=0.1):
        super(InputEmbedding, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.embedding(x) * math.sqrt(self.d_model)
        return self.dropout(x)
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len, dropout):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(seq_len, d_model) #matrix of shape (seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) #vector of shape (seq_len, 1)
        denominator = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * denominator)
        pe[:, 1::2] = torch.cos(position * denominator)
        pe = pe.unsqueeze(0) #shape = (1, seq_len, d_model)
        self.register_buffer('pe', pe)  # Register as a buffer to avoid being treated as a parameter

    def forward(self, x):
        # x = x + self.pe[:, :x.size(1)]
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # (batch, seq_len, d_model)
        return self.dropout(x)
    
class LayerNormalization(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # will be multiplied
        self.bias = nn.Parameter(torch.zeros(1)) # will be added to the normalized output
    
    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True)
        std = x.std(dim = -1, keepdim = True)

        return self.alpha * (x - mean) / (std + self.eps) + self.bias
    
class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        #(Batch, seq_len, d_model) -> (Batch, d_model, dff) -> (Batch, seq_len, d_model
        ffw = self.linear1(x)
        ffw = torch.relu(ffw)
        ffw = self.dropout(ffw)

        return self.linear2(ffw)
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        assert self.d_k * n_heads == d_model, "d_model must be divisible by n_heads"
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out_linear = nn.Linear(d_model, d_model)
    
    @staticmethod
    def attention(q, k, v, mask, dropout):
        d_k = q.size(-1)
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
          mask_value = -1e9
          if attention_scores.dtype == torch.float16:
              mask_value = -65504.0

          attention_scores = attention_scores.masked_fill(mask == 0, mask_value)
            # attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        attention_weights = torch.softmax(attention_scores, dim=-1) # (Batch, n_heads, seq_len, seq_len)
        if dropout is not None:
            attention_weights = dropout(attention_weights)
        return torch.matmul(attention_weights, v), attention_weights  # (Batch, n_heads, seq_len, d_k)

    def forward(self, query, key, value, mask):
        q = self.q_linear(query)  # (Batch, seq_len, d_model)
        k = self.k_linear(key)    # (Batch, seq_len, d_model)
        v = self.v_linear(value)  # (Batch, seq_len, d_model)

        q = q.view(q.shape[0], q.shape[1], self.n_heads, self.d_k).transpose(1, 2)  # (Batch, n_heads, seq_len, d_k)
        k = k.view(k.shape[0], k.shape[1], self.n_heads, self.d_k).transpose(1, 2)  # (Batch, n_heads, seq_len, d_k)
        v = v.view(v.shape[0], v.shape[1], self.n_heads, self.d_k).transpose(1, 2)  # (Batch, n_heads, seq_len, d_k)

        x, self.attention_scores = MultiHeadAttention.attention(q, k, v, mask, self.dropout) 
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.n_heads * self.d_k)  # (Batch, n_heads, seq_len, d_k) -> (Batch, seq_len, n_heads, d_k) -> (Batch, seq_len, d_model)
        return self.out_linear(x)  # (Batch, seq_len, d_model) 

class Residual(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sub_layer):
        return x + self.dropout(sub_layer(self.norm(x)))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout)
        self.residual1 = Residual(dropout)
        self.residual2 = Residual(dropout)

    def forward(self, x, mask):
        x = self.residual1(x, lambda x: self.self_attn(x, x, x, mask))
        return self.residual2(x, self.ffn)

class Encoder(nn.Module):
    def __init__(self, vocab_size, seq_len, d_model, n_heads, d_ff, num_layers, dropout=0.1):
        super().__init__()
        self.embedding = InputEmbedding(vocab_size, d_model, dropout)
        self.positional_encoding = PositionalEncoding(d_model, seq_len, dropout)  # Assuming max sequence length of 5000
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)])
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)  # Final normalization

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout)
        self.residual1 = Residual(dropout)
        self.residual2 = Residual(dropout)
        self.residual3 = Residual(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        x = self.residual1(x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.residual2(x, lambda x: self.cross_attn(x, enc_output, enc_output, src_mask))
        return self.residual3(x, self.ffn)

class Decoder(nn.Module):
    def __init__(self, vocab_size, seq_len, d_model, n_heads, d_ff, num_layers, dropout=0.1):
        super().__init__()
        self.embedding = InputEmbedding(vocab_size, d_model, dropout)
        self.positional_encoding = PositionalEncoding(d_model, seq_len, dropout)  # Assuming max sequence length of 5000
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)])
        self.norm = LayerNormalization()

    def forward(self, x, enc_output, src_mask, tgt_mask):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        return self.norm(x)  # Final normalization
    
class LinearLayer(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return torch.log_softmax(self.linear(x), dim = -1)  # (Batch, seq_len, d_model) -> (Batch, seq_len, vocab_size)
    
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, seq_len, d_model = 512, n_heads = 8, d_ff = 2048, num_layers = 6, dropout=0.1):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, seq_len, d_model, n_heads, d_ff, num_layers, dropout)
        self.decoder = Decoder(tgt_vocab_size, seq_len, d_model, n_heads, d_ff, num_layers, dropout)
        self.linear = LinearLayer(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_mask, tgt_mask):
        enc_output = self.encoder(src, src_mask)
        dec_output = self.decoder(tgt, enc_output, src_mask, tgt_mask)
        return self.linear(dec_output)  # (Batch, seq_len_tgt, vocab_size_tgt)