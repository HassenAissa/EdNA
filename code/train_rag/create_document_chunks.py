from pathlib import Path
import os
import gc

import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
tqdm.pandas()

# Configuration
CREATE_SMALL_VERSION = True  # Set to True to create 100k chunks version, False for full dataset
MAX_CHUNKS_PER_DATASET = 50_000 if CREATE_SMALL_VERSION else None
MAX_TOKENS = 512
TOKENIZER_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Paths
DATA_DIR = Path(__file__).parent / "data"
WIKI_CSV_PATH = DATA_DIR / "wiki_stem_corpus.csv"
MCQA_DATASET_NAME = "HAissa/MNLP_M3_mcqa_dataset"
OUTPUT_PATH = DATA_DIR / ("combined_dataset_100k.csv" if CREATE_SMALL_VERSION else "combined_dataset_full.csv")

# Memory management settings
CHUNK_SIZE = 10000   # Process in chunks of this size
MAX_COMBINED_ROWS = 100000  # Maximum rows in combined dataset


def safe_save_large_csv(df, output_path, chunk_size=CHUNK_SIZE):
    """Safely save large DataFrame to CSV with chunked writing"""
    print(f"Saving {len(df):,} rows to {output_path}")
    
    try:
        with tqdm(total=1, desc="Saving CSV") as pbar:
            df.to_csv(output_path, index=False)
            pbar.update(1)
            
    except MemoryError:
        print("Memory error during save! Trying chunked approach...")
        save_csv_in_chunks(df, output_path, chunk_size)
    except Exception as e:
        print(f"Error saving CSV: {e}")
        raise

def save_csv_in_chunks(df, output_path, chunk_size):
    """Save DataFrame to CSV in chunks to avoid memory issues"""
    print(f"Saving in chunks of {chunk_size:,} rows...")
    
    # Save header first
    df.head(0).to_csv(output_path, index=False)
    
    # Append chunks
    for i in tqdm(range(0, len(df), chunk_size), desc="Writing chunks"):
        chunk = df.iloc[i:i+chunk_size]
        chunk.to_csv(output_path, mode='a', header=False, index=False)
        
        # Force garbage collection between chunks
        gc.collect()

def count_tokens(text, tokenizer):
    """Count tokens in text using the specified tokenizer."""
    try:
        tokens = tokenizer.encode(text, add_special_tokens=False, truncation=False)
        return len(tokens)
    except Exception as e:
        print(f"Error tokenizing text: {e}")
        return float('inf')  # Return high value to exclude problematic texts

def count_tokens_and_truncate(text, tokenizer, max_length=512):
    """Count tokens and optionally truncate text to match token limit."""
    try:
        # First, get the full token count
        full_tokens = tokenizer.encode(text, add_special_tokens=False, truncation=False)
        full_token_count = len(full_tokens)
        
        if full_token_count <= max_length:
            # Text is already within limit
            return text, full_token_count
        else:
            # Truncate both tokens and decode back to text
            truncated_tokens = tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=max_length)
            truncated_text = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
            return truncated_text, len(truncated_tokens)
    except Exception as e:
        print(f"Error processing text (length: {len(text)}): {e}")
        return text, np.nan

def process_wiki_dataset():
    """Load and process wiki STEM corpus dataset."""
    print("Loading wiki STEM corpus...")
    
    if not WIKI_CSV_PATH.exists():
        raise FileNotFoundError(f"Wiki corpus file not found: {WIKI_CSV_PATH}")
    
    wiki_df = pd.read_csv(WIKI_CSV_PATH)
    print(f"Loaded {len(wiki_df):,} wiki entries")
    
    # Check required columns
    required_cols = ['page_title', 'section_title', 'breadcrumb', 'text']
    missing_cols = [col for col in required_cols if col not in wiki_df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in wiki dataset: {missing_cols}")
    
    print("Processing wiki entries...")
    processed_texts = []
    
    for _, row in tqdm(wiki_df.iterrows(), total=len(wiki_df), desc="Processing wiki texts"):
        # Merge columns with HTML-like tags
        page_title = str(row['page_title']) if pd.notna(row['page_title']) else ""
        section_title = str(row['section_title']) if pd.notna(row['section_title']) else ""
        breadcrumb = str(row['breadcrumb']) if pd.notna(row['breadcrumb']) else ""
        text = str(row['text']) if pd.notna(row['text']) else ""
        
        # Create merged text with tags
        merged_text = ""
        if page_title:
            merged_text += f"<page_title>{page_title}</page_title>\n"
        if section_title:
            merged_text += f"<section_title>{section_title}</section_title>\n"
        if breadcrumb:
            merged_text += f"<breadcrumb>{breadcrumb}</breadcrumb>\n"
        if text:
            merged_text += f"<text>{text}</text>"
        
        processed_texts.append(merged_text.strip())
    
    # Create processed dataframe
    wiki_processed = pd.DataFrame({
        'text': processed_texts,
        'source': 'https://www.kaggle.com/datasets/conjuring92/wiki-stem-corpus'
    })
    
    print(f"Processed {len(wiki_processed):,} wiki entries")
    return wiki_processed

def process_mcqa_dataset():
    """Load and process MCQA dataset from HuggingFace."""
    print("Loading MCQA dataset from HuggingFace...")
    
    dataset = load_dataset(MCQA_DATASET_NAME, split="train")
    print(f"Loaded {len(dataset):,} MCQA entries")
    
    # Check required columns
    if 'question' not in dataset.column_names or 'answer' not in dataset.column_names:
        raise ValueError(f"Missing required columns. Available: {dataset.column_names}")
    
    print("Processing MCQA entries...")
    processed_data = []
    
    for example in tqdm(dataset, desc="Processing MCQA texts"):
        # Create text from question + answer
        question = str(example['question']) if example['question'] else ""
        answer = str(example['answer']) if example['answer'] else ""
        text = f"{question}\n{answer}".strip()
        
        # Get source or use default
        source = example.get('source', 'HAissa/MNLP_M3_mcqa_dataset')
        
        if text:  # Only add if text is not empty
            processed_data.append({
                'text': text,
                'source': source
            })
    
    mcqa_processed = pd.DataFrame(processed_data)
    print(f"Processed {len(mcqa_processed):,} MCQA entries")
    return mcqa_processed

def filter_by_tokens(df, tokenizer, max_chunks=None, apply_token_filter=True):
    """Filter dataset by token count and optionally limit number of chunks."""
    if apply_token_filter:
        print(f"Filtering dataset by token count (max {MAX_TOKENS} tokens)...")
        
        # Add token counts
        token_counts = []
        for text in tqdm(df['text'], desc="Counting tokens"):
            token_count = count_tokens(text, tokenizer)
            token_counts.append(token_count)
        
        df['token_count'] = token_counts
        
        # Filter by token limit
        filtered_df = df[df['token_count'] <= MAX_TOKENS].copy()
        print(f"After token filtering: {len(filtered_df):,} entries (removed {len(df) - len(filtered_df):,})")
        
        if max_chunks and len(filtered_df) > max_chunks:
            # Sort by token count (descending) to get entries closest to 512 tokens
            filtered_df = filtered_df.sort_values('token_count', ascending=False)
            filtered_df = filtered_df.head(max_chunks)
            print(f"Limited to top {max_chunks:,} entries with highest token counts")
    else:
        print("No token filtering applied - using full dataset")
        filtered_df = df.copy()
        
        # Still add token counts for statistics if requested
        if max_chunks is None:  # Only compute token stats for full dataset
            print("Computing token statistics...")
            token_counts = []
            for text in tqdm(df['text'], desc="Counting tokens for statistics"):
                token_count = count_tokens(text, tokenizer)
                token_counts.append(token_count)
            filtered_df['token_count'] = token_counts
    
    return filtered_df

def main():
    print("=== Document Chunks Creation ===")
    print(f"Small version: {CREATE_SMALL_VERSION}")
    if CREATE_SMALL_VERSION:
        print(f"Target: {MAX_CHUNKS_PER_DATASET * 2:,} total chunks ({MAX_CHUNKS_PER_DATASET:,} from each dataset)")
        print(f"Token limit: {MAX_TOKENS} tokens per chunk")
    else:
        print("Full dataset mode - no token limits or chunk restrictions")
    
    # Initialize tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL_NAME)
    
    # Process datasets
    wiki_df = process_wiki_dataset()
    mcqa_df = process_mcqa_dataset()
    
    # Filter by tokens and limit chunks if needed
    if CREATE_SMALL_VERSION:
        # Small version: apply token filter and limit chunks
        wiki_filtered = filter_by_tokens(wiki_df, tokenizer, MAX_CHUNKS_PER_DATASET, apply_token_filter=True)
        mcqa_filtered = filter_by_tokens(mcqa_df, tokenizer, MAX_CHUNKS_PER_DATASET, apply_token_filter=True)
    else:
        # Full version: no restrictions
        wiki_filtered = filter_by_tokens(wiki_df, tokenizer, apply_token_filter=False)
        mcqa_filtered = filter_by_tokens(mcqa_df, tokenizer, apply_token_filter=False)
    
    # Combine datasets
    print("Combining datasets...")
    combined_df = pd.concat([wiki_filtered, mcqa_filtered], ignore_index=True)
    
    # Remove token_count column for final output
    final_df = combined_df[['text', 'source']].copy()
    
    print(f"Final combined dataset: {len(final_df):,} entries")
    print(f"  - Wiki entries: {len(wiki_filtered):,}")
    print(f"  - MCQA entries: {len(mcqa_filtered):,}")
    
    # Save combined dataset
    print(f"Saving to {OUTPUT_PATH}...")
    DATA_DIR.mkdir(exist_ok=True)
    final_df.to_csv(OUTPUT_PATH, index=False)
    
    print(f"✅ Successfully saved combined dataset to {OUTPUT_PATH}")
    
    # Print statistics
    if 'token_count' in combined_df.columns:
        avg_tokens = combined_df['token_count'].mean()
        max_tokens = combined_df['token_count'].max()
        min_tokens = combined_df['token_count'].min()
        print(f"Token statistics: avg={avg_tokens:.1f}, min={min_tokens}, max={max_tokens}")
        
if __name__ == "__main__":
    main()