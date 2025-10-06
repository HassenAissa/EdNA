# RAG Training Pipeline

This folder contains the complete pipeline for training and using a Retrieval-Augmented Generation (RAG) system for multiple-choice question answering in advanced STEM courses.

## Overview

The RAG system combines document retrieval with text generation to answer questions by:
1. Creating document chunks from Wikipedia STEM corpus and MCQA datasets
2. Building a FAISS vector index for efficient retrieval
3. Training a generator model using the RAFT (Retrieval Augmented Fine-Tuning) approach
4. Providing both command-line and GUI interfaces for RAG inference

## Files

### Core Scripts

- **`create_document_chunks.py`** - Processes and combines datasets
  - Loads Wikipedia STEM corpus and MCQA dataset from HuggingFace
  - Applies token filtering (less than 512 tokens) and creates balanced 100k dataset (50k from each source)
  - Outputs: `combined_dataset_100k.csv`

- **`run.py`** - Command-line RAG inference system
  - Loads all models and implement the RAG system
  - Automatically creates FAISS index from combined dataset if not found

- **`gui_viewer.py`** - GUI interface for RAG testing
  - Shows RAG answer, base model answer, retrieved documents, and complete prompt to allow qualitative comparaison between with and without RAG

- **`create_raft_dataset.py`** - Creates RAFT training dataset
  - Creates the dataset for RAFT training by combining Q&A pairs with retrieved golden and distractor document pairs

- **`train_generator.py`** - Finetunes generator model using RAFT
  - Implements DoRA (Decomposed Rank Adaptation) for efficient training on RAFT dataset

## Usage

### Prerequisites

1. **Download Wikipedia STEM Corpus**: Get the dataset from [Kaggle](https://www.kaggle.com/datasets/conjuring92/wiki-stem-corpus) and place `wiki_stem_corpus.csv` in the `data/` folder.

2. **Install Dependencies**: 
   ```bash
   pip install -r requirements_rag.txt
   ```

### Quick Start

1. **Create Document Chunks**:
   ```bash
   python create_document_chunks.py
   ```
   This creates a balanced 100k dataset from Wikipedia STEM and MCQA sources.

2. **Run RAG System** (Command-line):
   ```bash
   python run.py
   ```
   All models load automatically. The system will create FAISS index if needed.

### Full pipeline

#### RAFT Training Pipeline

1. **Create RAFT Dataset**:
   ```bash
   python create_raft_dataset.py
   ```
   Requires existing FAISS index and processes Q&A pairs with retrieved documents.

2. **Train Generator Model**:
   ```bash
   python train_generator.py
   ```
   Fine-tunes the generator using DoRA on the RAFT dataset.

#### Configuration Options

**Document Processing** (`create_document_chunks.py`):
- `CREATE_SMALL_VERSION = True`: Creates 100k balanced dataset
- `MAX_TOKENS = 512`: Token limit per chunk
- `MAX_CHUNKS_PER_DATASET = 50_000`: Maximum chunks per source

**RAG System** (`run.py`):
- `TOP_K_DOCS = 3`: Number of documents to retrieve
- `OUTPUT_MAX_LENGTH = 512`: Maximum tokens in generated answer
- `EMBEDDING_BATCH_SIZE = 32`: Batch size for embedding computation

## Models

The pipeline uses these pre-trained models:
- **Document Encoder**: `Lysandrec/MNLP_M3_document_encoder`
- **Generator Model**: `Lysandrec/MNLP_M3_rag_model`
- **Base Q&A Model**: `HAissa/MNLP_M3_mcqa_model` - Base model for comparison in `gui_viewer.py`