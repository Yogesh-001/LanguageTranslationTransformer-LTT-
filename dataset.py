import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):
    def __init__(self, data, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        self.data = data
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.seq_len = seq_len

        self.sos_token = torch.tensor([tokenizer_src.token_to_id("<SOS>")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_src.token_to_id("<EOS>")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_src.token_to_id("<PAD>")], dtype=torch.int64)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        src_target = self.data[index]
        src_text = src_target[self.src_lang]
        tgt_text = src_target[self.tgt_lang]

        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2  # -2 for [SOS] and [EOS]
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Input sequence is too long for the specified sequence length.")
        
        #Adding SOS and EOS tokens to encoder input
        encoder_input = torch.cat([
            self.sos_token,
            torch.tensor(enc_input_tokens, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64)
        ])

        #Adding only SOS because the input will start from SOS and the transformer will predict the next.
        decoder_input = torch.cat([
            self.sos_token,
            torch.tensor(dec_input_tokens, dtype = torch.int64),
            torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype = torch.int64)
        ])

        label = torch.cat([
            torch.tensor(dec_input_tokens, dtype = torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype = torch.int64)
        ])

        assert encoder_input.size(0) == self.seq_len, f"Encoder input length {encoder_input.size(0)} does not match seq_len {self.seq_len}"
        assert decoder_input.size(0) == self.seq_len, f"Decoder input length {decoder_input.size(0)} does not match seq_len {self.seq_len}"
        assert label.size(0) == self.seq_len, f"Label length {label.size(0)} does not match seq_len {self.seq_len}"

        return {
            'encoder_input': encoder_input,
            'decoder_input': decoder_input,
            'encoder_mask' : (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),  # (1, 1, seq_len)
            'decoder_mask' : (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),
            'label' : label,
            'src_text': src_text,
            'tgt_text' : tgt_text,
        }
    
# we want each word in the decoder to only attend to the previous words and not the future words.
def causal_mask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0