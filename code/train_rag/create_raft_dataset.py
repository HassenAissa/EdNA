import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from langchain_core.documents import Document as LCDocument
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os
import torch
from pathlib import Path

# Configuration
EMBEDDING_MODEL = "Lysandrec/MNLP_M3_document_encoder"

DATA_DIR = Path(__file__).parent / "data"
QA_DATASET_ID = "HAissa/MNLP_M3_mcqa_dataset"
COMBINED_DATASET_PATH = DATA_DIR / "combined_dataset_100k.csv"
FAISS_INDEX_PATH = DATA_DIR / "faiss_index"
OUTPUT_FILE = DATA_DIR / "raft_training_data.csv"

TOP_K_RETRIEVE = 5
BATCH_SAVE_SIZE = 10000

# Device setup
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("Using Apple Silicon GPU (MPS)")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU")

def create_faiss_index_from_csv(csv_path, embedding_function, index_path):
    """Create FAISS index from CSV file containing document chunks."""
    print(f"Creating FAISS index from {csv_path}...")
    
    # Load the combined dataset
    if not csv_path.exists():
        raise FileNotFoundError(f"Combined dataset not found at {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} documents from CSV")
    
    # Convert to LangChain documents
    documents = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Converting to documents"):
        # Create metadata
        metadata = {
            'source': row['source'],
            'doc_id': idx,
            'char_length': len(row['text'])
        }
        
        # Create LangChain document
        doc = LCDocument(
            page_content=row['text'],
            metadata=metadata
        )
        documents.append(doc)
    
    print(f"Created {len(documents):,} LangChain documents")
    
    # Create FAISS vector store
    print("Creating FAISS vector store (this may take a while)...")
    vector_store = FAISS.from_documents(
        documents=documents,
        embedding=embedding_function
    )
    
    # Save the index
    print(f"Saving FAISS index to {index_path}...")
    index_path.parent.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(index_path)
    
    print(f"✅ FAISS index created and saved successfully!")
    return vector_store

def load_or_create_faiss_index(embedding_function):
    """Load existing FAISS index or create new one if it doesn't exist."""
    if FAISS_INDEX_PATH.exists():
        print(f"Loading existing FAISS index from {FAISS_INDEX_PATH}...")
        vector_store = FAISS.load_local(
            FAISS_INDEX_PATH, 
            embedding_function, 
            allow_dangerous_deserialization=True
        )
        print("FAISS index loaded successfully")
        return vector_store
    else:
        print(f"FAISS index not found at {FAISS_INDEX_PATH}")
        print("Creating new FAISS index from combined dataset...")
        return create_faiss_index_from_csv(
            COMBINED_DATASET_PATH, 
            embedding_function, 
            FAISS_INDEX_PATH
        )

def save_batch_to_csv(samples_batch, output_file, write_header=False):
    """Efficiently save a batch of samples to CSV"""
    if not samples_batch:
        return
    
    df_batch = pd.DataFrame(samples_batch)
    df_batch.to_csv(output_file, mode='a', header=write_header, index=False)
    print(f"Saved batch of {len(samples_batch)} samples to {output_file}")

def main():
    print("=== RAFT Dataset Creation ===")
    
    print("Loading Q/A dataset...")
    qa_dataset = load_dataset(QA_DATASET_ID, split="train")
    print(f"Loaded {len(qa_dataset)} Q/A pairs")

    print("Initializing embedding function...")
    embedding_function = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL, 
        model_kwargs={'device': DEVICE}
    )

    # Load or create FAISS index
    vector_store = load_or_create_faiss_index(embedding_function)

    print("Creating RAFT dataset...")
    
    # Initialize CSV file with headers only if file doesn't exist
    file_exists = OUTPUT_FILE.exists()
    if not file_exists:
        # Create empty file with headers
        header_df = pd.DataFrame(columns=[
            "question", "golden_document", 
            "distractor_documents", "chain_of_thought_answer",
            "source"
        ])
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        header_df.to_csv(OUTPUT_FILE, index=False)
        print(f"Created new CSV file: {OUTPUT_FILE}")
    else:
        print(f"Appending to existing CSV file: {OUTPUT_FILE}")
    
    samples_batch = []
    samples_count = 0
    
    for qa_pair in tqdm(qa_dataset, desc="Processing Q/A pairs"):
        question = qa_pair.get("question")
        answer = qa_pair.get("answer")
        source = qa_pair.get("source")
        
        if not question or not answer:
            continue
            
        # Retrieve top 5 documents using FAISS vector store
        try:
            docs_with_scores = vector_store.similarity_search_with_score(question, k=TOP_K_RETRIEVE)
        except Exception as e:
            print(f"Error during retrieval for question: {question[:50]}... Error: {e}")
            continue
        
        if len(docs_with_scores) < 5:
            continue  # Skip if we don't have enough documents
            
        # Golden document is the 1st retrieved
        golden_document = docs_with_scores[0][0].page_content
        
        # Distractor document is the 5th retrieved (index 4)
        distractor_document = docs_with_scores[4][0].page_content
        
        raft_sample = {
            "question": question,
            "golden_document": golden_document,
            "distractor_documents": distractor_document,
            "chain_of_thought_answer": answer,
            "source": source
        }
        
        # Add to batch
        samples_batch.append(raft_sample)
        samples_count += 1
        
        # Save batch when reaching BATCH_SAVE_SIZE
        if len(samples_batch) >= BATCH_SAVE_SIZE:
            save_batch_to_csv(samples_batch, OUTPUT_FILE, write_header=False)
            samples_batch = []  # Clear the batch
    
    # Save any remaining samples in the final batch
    if samples_batch:
        save_batch_to_csv(samples_batch, OUTPUT_FILE, write_header=False)

    print(f"Generated {samples_count} RAFT samples")
    print(f"Saved RAFT dataset to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()