from pathlib import Path

def get_config():
    return {
        "batch_size": 8,
        "num_epochs": 10,
        "lr" : 10**-4,
        "seq_len": 512,
        "d_model" : 512,
        "n_heads": 8,
        "d_ff": 2048,
        "num_layers": 6,
        "dropout": 0.1,
        "lang_src" : "Input", #english for the given dataset
        "lang_tgt" : "Output", #telugu
        "model_folder" : "weights",
        "model_filename" : "transformer_model_",
        "preload" : None,
        "tokenizer_file" : "tokenizer_{0}.json",
        "experimental_name" : "runs/transformer_model_"
    }


def get_weights_file_path(config, epoch=None):
    model_folder = config['model_folder']
    model_basename = config['model_filename']
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

# Find the latest weights file in the weights folder
def latest_weights_file_path(config):
    model_folder = f"{config['model_folder']}"
    model_filename = f"{config['model_filename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])