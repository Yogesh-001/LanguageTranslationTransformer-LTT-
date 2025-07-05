from pathlib import Path
from config import get_config, latest_weights_file_path 
from model import Transformer
from tokenizers import Tokenizer
from datasets import load_dataset
from dataset import BilingualDataset, causal_mask
import torch
import sys

def translate(sentence: str):
    # Define the device, tokenizers, and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    config = get_config()
    tokenizer_src = Tokenizer.from_file(str(Path(config['tokenizer_file'].format(config['lang_src']))))
    tokenizer_tgt = Tokenizer.from_file(str(Path(config['tokenizer_file'].format(config['lang_tgt']))))
    model = Transformer(tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size(), seq_len=config['seq_len'], d_model=config['d_model']).to(device)

    # Load the pretrained weights
    model_filename = latest_weights_file_path(config)
    state = torch.load(model_filename, map_location=device)
    # print("Loading weights from:", model_filename)
    # print("Model keys expected:")
    # for k in model.state_dict().keys():
    #     print("   ", k)
    cleaned_state_dict = {
      k.replace('_orig_mod.', ''): v for k, v in state['model_state_dict'].items()
    } 
    print("Checkpoint keys:", cleaned_state_dict.keys())
    # print("Model keys in checkpoint:")
    # for k in cleaned_state_dict.keys():
    #     print("   ", k)
    # model.load_state_dict(state['model_state_dict'])
    # Load the cleaned state dict
    model.load_state_dict(cleaned_state_dict)

    # if the sentence is a number use it as an index to the test set
    label = ""
    if type(sentence) == int or sentence.isdigit():
        id = int(sentence)
        ds = load_dataset(f"{config['datasource']}", f"{config['lang_src']}-{config['lang_tgt']}", split='all')
        ds = BilingualDataset(ds, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
        sentence = ds[id]['src_text']
        label = ds[id]["tgt_text"]
    seq_len = config['seq_len']

    # translate the sentence
    model.eval()
    with torch.no_grad():
        # Precompute the encoder output and reuse it for every generation step
        source = tokenizer_src.encode(sentence)
        source = torch.cat([
            torch.tensor([tokenizer_src.token_to_id('<SOS>')], dtype=torch.int64), 
            torch.tensor(source.ids, dtype=torch.int64),
            torch.tensor([tokenizer_src.token_to_id('<EOS>')], dtype=torch.int64),
            torch.tensor([tokenizer_src.token_to_id('<PAD>')] * (seq_len - len(source.ids) - 2), dtype=torch.int64)
        ], dim=0).to(device)
        source_mask = (source != tokenizer_src.token_to_id('<PAD>')).unsqueeze(0).unsqueeze(0).int().to(device)

        # Initialize the decoder input with the sos token
        decoder_input = torch.empty(1, 1).fill_(tokenizer_tgt.token_to_id('<SOS>')).type_as(source).to(device)

        # Print the source sentence and target start prompt
        if label != "": print(f"{f'ID: ':>12}{id}") 
        print(f"{f'SOURCE: ':>12}{sentence}")
        if label != "": print(f"{f'TARGET: ':>12}{label}") 
        print(f"{f'PREDICTED: ':>12}", end='')

        # Generate the translation word by word
        while decoder_input.size(1) < seq_len:

            decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
            out = model(source, decoder_input, source_mask, decoder_mask)  # (B, seq_len, target_vocab_size)

            out = out[:, -1, :]  # Get the last token's output

            _, next_word = torch.max(out, dim=1)  # Get the index of the max logit

            decoder_input = torch.cat([decoder_input, torch.empty(1,1).type_as(source).fill_(next_word.item()).to(device)], dim=1)  # Append the next word to the decoder input

            # print the translated word
            print(f"{tokenizer_tgt.decode([next_word.item()])}", end=' ')

            # break if we predict the end of sentence token
            if next_word == tokenizer_tgt.token_to_id('[EOS]'):
                break

    # convert ids to tokens
    return tokenizer_tgt.decode(decoder_input[0].tolist())
    
#read sentence from argument
translate(sys.argv[1] if len(sys.argv) > 1 else "I am not a very good a student.")