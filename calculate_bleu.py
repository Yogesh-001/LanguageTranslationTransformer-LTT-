import torch
import random
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from translate import translate
from config import get_config
from tqdm import tqdm
import time
import argparse

# Download NLTK BLEU score packages if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def load_test_data(file_path, num_samples=100):
    """Load test data from the specified file."""
    print(f"Loading test data from {file_path}")
    
    # Read all lines from the file
    with open(file_path, 'r', encoding='utf-8') as f:
        all_lines = f.readlines()
    
    # Filter out empty lines
    data_pairs = [line.strip().split('++++$++++') for line in all_lines if line.strip()]
    data_pairs = [(src, tgt) for src, tgt in data_pairs if len(src) > 0 and len(tgt) > 0]
    
    # If we have fewer pairs than requested samples, use all pairs
    if len(data_pairs) <= num_samples:
        test_pairs = data_pairs
    else:
        # Use random.sample to select a random subset
        test_pairs = random.sample(data_pairs, num_samples)
    
    print(f"Selected {len(test_pairs)} test samples out of {len(data_pairs)} total pairs")
    return test_pairs

def calculate_bleu(test_pairs, force_cpu=False, max_length=50, verbose=False):
    """Calculate BLEU score for the test pairs."""
    references = []
    hypotheses = []
    
    # Set up for BLEU score calculation
    smoothing = SmoothingFunction().method1
    
    print(f"Translating {len(test_pairs)} sentences...")
    start_time = time.time()
    
    # Process each test pair
    for i, (source, reference) in enumerate(tqdm(test_pairs)):
        # Tokenize reference
        reference_tokens = nltk.word_tokenize(reference)
        references.append([reference_tokens])
        
        # Translate source text
        try:
            translation = translate(source, force_cpu=force_cpu, max_length=max_length)
            translation_tokens = nltk.word_tokenize(translation)
            
            # Some basic checks
            if not translation_tokens:
                translation_tokens = ['']  # Empty translation
                
            hypotheses.append(translation_tokens)
            
            if verbose and i < 10:  # Show first 10 examples
                print(f"\nSource: {source}")
                print(f"Reference: {reference}")
                print(f"Hypothesis: {translation}")
                
        except Exception as e:
            print(f"Error translating: {source}")
            print(f"Error: {str(e)}")
            # Add a placeholder for failed translations
            hypotheses.append([''])
    
    # Calculate BLEU scores
    print("\nCalculating BLEU scores...")
    bleu1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0), smoothing_function=smoothing)
    bleu2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
    bleu3 = corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing)
    bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
    
    elapsed_time = time.time() - start_time
    
    return {
        'bleu1': bleu1 * 100,
        'bleu2': bleu2 * 100,
        'bleu3': bleu3 * 100,
        'bleu4': bleu4 * 100,
        'time': elapsed_time,
        'samples': len(test_pairs)
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate BLEU score for the transformer translation model')
    parser.add_argument('--samples', type=int, default=100, help='Number of test samples to use')
    parser.add_argument('--cpu', action='store_true', help='Force CPU usage')
    parser.add_argument('--max-length', type=int, default=50, help='Maximum translation length')
    parser.add_argument('--verbose', action='store_true', help='Show detailed output')
    args = parser.parse_args()
    
    # Load configuration and test data
    config = get_config()
    test_data_file = "english_telugu_data.txt"
    test_pairs = load_test_data(test_data_file, args.samples)
    
    print(f"Calculating BLEU score for {len(test_pairs)} test pairs")
    print(f"Using {'CPU' if args.cpu else 'GPU if available'} for translation")
    
    # Calculate BLEU score
    results = calculate_bleu(test_pairs, force_cpu=args.cpu, max_length=args.max_length, verbose=args.verbose)
    
    # Print results
    print("\n========= BLEU Score Results =========")
    print(f"BLEU-1: {results['bleu1']:.2f}%")
    print(f"BLEU-2: {results['bleu2']:.2f}%")
    print(f"BLEU-3: {results['bleu3']:.2f}%")
    print(f"BLEU-4: {results['bleu4']:.2f}%")
    print(f"Time taken: {results['time']:.2f} seconds for {results['samples']} samples")
    print(f"Average time per translation: {results['time']/results['samples']:.2f} seconds")
    print("=======================================")
