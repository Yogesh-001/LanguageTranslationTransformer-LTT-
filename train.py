import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from config import get_config, get_weights_file_path
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from dataset import BilingualDataset, causal_mask
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from model import Transformer
from tqdm import tqdm
from torch.amp import GradScaler, autocast

def greedY_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('<SOS>')
    eos_idx = tokenizer_tgt.token_to_id('<EOS>')

    #precompute the encoder output and reuse it for every token we get for the decoder.
    decoder_input = torch.empty(1,1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break
        
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        out = model(source, decoder_input, source_mask, decoder_mask)  # (B, seq_len, target_vocab_size)

        out = out[:, -1, :]  # Get the last token's output

        _, next_word = torch.max(out, dim=1)  # Get the index of the max logit

        decoder_input = torch.cat([decoder_input, torch.empty(1,1).type_as(source).fill_(next_word.item()).to(device)], dim=1)  # Append the next word to the decoder input
        if next_word == eos_idx:
            break
    
    return decoder_input.squeeze(0)

def run_validation(model, validation_data, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_state, writer, num_examples = 2):
    model.eval()
    count = 0

    source_txts = []
    expected = []
    predicted = []

    console_width = 0
    with torch.no_grad():
        for batch in validation_data:
            count += 1
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)

            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            model_out  = greedY_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)
            
            source_txt = batch['src_text'][0]
            target_txt = batch['tgt_text'][0]
            predicted_txt = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_txts.append(source_txt)
            expected.append(target_txt)
            predicted.append(predicted_txt)

            #TQDM print for the console.
            print_msg('----'*console_width)
            print_msg(f'SOURCE: {source_txt}')
            print_msg(f'TARGET: {target_txt}')
            print_msg(f'PREDICTED: {predicted_txt}')

            if count == max_len:
                break
        



def get_all_sentences(ds, lang):
    for item in ds:
        yield item[lang]

def check_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        print(f"Tokenizer for {lang} not found. Training a new tokenizer...")
        tokenizer = Tokenizer(WordLevel(unk_token="<UNK>"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["<UNK>", "<PAD>", "<SOS>", "<EOS>"])
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_dataset(config):
    dataset = load_dataset("MRR24/English_to_Telugu_Bilingual_Sentence_Pairs", split='train')
    dataset = dataset.shuffle(seed=42).select(range(40000))
    tokenizer_src = check_tokenizer(config, dataset, config['lang_src'])
    tokenizer_tgt = check_tokenizer(config, dataset, config['lang_tgt'])

    #split the dataset
    train_split = int(len(dataset) * 0.9)
    valid_split = len(dataset) - train_split

    train_data, valid_data = random_split(dataset, [train_split, valid_split])

    train_dataset = BilingualDataset(train_data, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    valid_dataset = BilingualDataset(valid_data, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    max_len_src = 0
    max_len_tgt = 0

    for item in dataset:
        src_ids = tokenizer_src.encode(item[config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item[config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f"Max length for source language ({config['lang_src']}): {max_len_src}")
    print(f"Max length for target language ({config['lang_tgt']}): {max_len_tgt}")

    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    val_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False, pin_memory=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_size_src, vocab_size_tgt):
    model = Transformer(
        src_vocab_size=vocab_size_src,
        tgt_vocab_size=vocab_size_tgt,
        seq_len=config['seq_len'],
        d_model=config['d_model'],
    )
    return model

def train_model(config):
    # CPU or GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    print(f"Using device: {device}")

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_dataset(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    model = torch.compile(model, mode='max-autotune', fullgraph=True)  # Compile the model for better performance
    writer  = SummaryWriter(config['experimental_name'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)
    
    # adding a mixed precision scaler
    scaler = GradScaler()

    inital_epoch = 0
    global_step = 0
    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f'preloading model: {model_filename}')
        state = torch.load(model_filename)
        inital_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_tgt.token_to_id("<PAD>"), label_smoothing=0.1).to(device)
    for epoch in range(inital_epoch, config['num_epochs']):
        batch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config['num_epochs']}", unit="batch")
        for batch in batch_iterator:
            model.train()
            encoder_input = batch['encoder_input'].to(device)  #(B,seq_len)
            decoder_input = batch['decoder_input'].to(device)  #(B,seq_len)
            encoder_mask = batch['encoder_mask'].to(device)    #(B,1,1,seq_len)
            decoder_mask = batch['decoder_mask'].to(device)    #(B,1,seq_len,seq_len)
            label =  batch['label'].to(device) #(B,seq_len)

            with autocast(device_type='cuda'):
                outputs = model(encoder_input, decoder_input, encoder_mask,  decoder_mask)  #(B, seq_len, target_vocab_size)
                loss = loss_fn(outputs.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
    
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})
            writer.add_scalar('Loss/train', loss.item(), global_step)
            writer.flush()

            # loss.backward()
            # optimizer.step()
            # Backprop with AMP scaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            global_step += 1
        
        run_validation(model, val_dataloader,tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg:batch_iterator.write(msg), global_step, writer)
        model_filename = get_weights_file_path(config, f'{epoch:02d}')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)
        # # If model = torch.compile(...)
        # torch.save({
        #     'epoch': epoch,
        #     'model_state_dict': model._orig_mod.state_dict(),  # no extra prefix
        #     'optimizer_state_dict': optimizer.state_dict(),
        #     'global_step': global_step
        # }, model_filename)


if __name__ == '__main__':
    config = get_config()
    train_model(config)