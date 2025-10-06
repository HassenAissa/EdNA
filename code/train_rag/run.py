import torch
import pandas as pd
import logging
from typing import List, Dict, Any
from tqdm import tqdm
from multiprocessing import cpu_count
import threading
from pathlib import Path

from langchain_core.documents import Document as LCDocument
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline as hf_pipeline, TextIteratorStreamer
from datasets import load_dataset

# Configuration
EMBEDDING_MODEL = "Lysandrec/MNLP_M3_document_encoder"
GENERATOR_MODEL = "Lysandrec/MNLP_M3_rag_model"
RAG_DOCUMENTS_REPO = "Lysandrec/MNLP_M3_rag_documents"
TOP_K_DOCS = 3
OUTPUT_MAX_LENGTH = 512

# Parallel processing configuration
MAX_WORKERS = min(cpu_count(), 8)
USE_MULTIPROCESSING = True
EMBEDDING_BATCH_SIZE = 32

# Paths
DATA_DIR = Path(__file__).parent / "data"
COMBINED_DATASET_PATH = DATA_DIR / "combined_dataset_100k.csv"
FAISS_INDEX_PATH = DATA_DIR / "faiss_index"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Device setup
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    logger.info("Using Apple Silicon GPU (MPS)")
else:
    DEVICE = torch.device("cpu")
    logger.info("Using CPU")

def process_csv_chunk(args):
    """Helper function for parallel CSV chunk processing"""
    chunk, text_column, chunk_idx = args
    documents = []
    
    for idx, row in chunk.iterrows():
        content = str(row[text_column]).strip()
        
        # Skip empty content
        if not content or content.lower() in ['nan', 'null', '']:
            continue
        
        metadata = {col: str(row[col]) for col in chunk.columns if col != text_column}
        metadata['row_id'] = idx
        metadata['chunk_id'] = chunk_idx
        
        doc = LCDocument(page_content=content, metadata=metadata)
        documents.append(doc)
    
    return documents

def process_dataset_batch(args):
    """Helper function for parallel dataset batch processing"""
    batch, text_column, dataset_id, start_idx = args
    documents = []
    
    for local_idx, record in enumerate(batch):
        content = str(record[text_column]).strip()
        
        # Skip empty content
        if not content or content.lower() in ['nan', 'null', '']:
            continue
        
        # Create metadata from other columns
        metadata = {col: str(record[col]) for col in record.keys() if col != text_column}
        metadata['row_id'] = start_idx + local_idx
        metadata['source'] = 'huggingface'
        metadata['dataset_id'] = dataset_id
        
        doc = LCDocument(page_content=content, metadata=metadata)
        documents.append(doc)
    
    return documents

class RAG:
    def __init__(self, embedding_model_name: str = EMBEDDING_MODEL, generator_model_name: str = GENERATOR_MODEL):
        """Initialize the RAG system with all models loaded upfront"""
        logger.info("Initializing RAG system...")
        
        # Initialize embeddings
        logger.info(f"Loading embedding model: {embedding_model_name}")
        self.embedding_function = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={'device': DEVICE}
        )
        logger.info("Embedding model loaded successfully")
        
        # Load generator model immediately
        logger.info(f"Loading generator model: {generator_model_name}")
        self.generator_model_name = generator_model_name
        self.generator_model = None
        self.generator_tokenizer = None
        self._generator_loaded = False
        
        # Load generator immediately
        self._load_generator()
        
        # Initialize vector store
        self.vector_store = None
        
        logger.info("RAG system initialized successfully - all models loaded")
    
    def _load_generator(self):
        """Load the generator model immediately"""
        try:
            self.generator_tokenizer = AutoTokenizer.from_pretrained(self.generator_model_name, padding_side='left')
            if self.generator_tokenizer.pad_token is None:
                self.generator_tokenizer.pad_token = self.generator_tokenizer.eos_token
                
            self.generator_model = AutoModelForCausalLM.from_pretrained(
                self.generator_model_name, 
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            self._generator_loaded = True
            logger.info("Generator model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading generator: {e}")
            logger.warning("Continuing without generator - will only do retrieval")
            self._generator_loaded = False
    
    def load_documents_from_huggingface(self, repo_id: str = RAG_DOCUMENTS_REPO) -> List[LCDocument]:
        """Load documents from HuggingFace dataset"""
        logger.info(f"Loading documents from HuggingFace repository: {repo_id}...")
        
        try:
            # Load dataset from HuggingFace
            dataset = load_dataset(repo_id, split="train")
            logger.info(f"Loaded {len(dataset):,} entries from HuggingFace")
            
            documents = []
            for idx, row in enumerate(tqdm(dataset, desc="Converting to documents")):
                # Assuming the dataset has 'text' and 'source' columns
                # Adjust these column names if they're different in your dataset
                text_content = row.get('text', '')
                source = row.get('source', repo_id)
                
                if not text_content or text_content.strip() == '':
                    continue
                
                metadata = {
                    'source': source,
                    'doc_id': idx,
                    'char_length': len(text_content)
                }
                
                # Add any other metadata fields present in the dataset
                for key, value in row.items():
                    if key not in ['text', 'source']:
                        metadata[key] = str(value)
                
                doc = LCDocument(
                    page_content=text_content,
                    metadata=metadata
                )
                documents.append(doc)
            
            logger.info(f"Created {len(documents):,} LangChain documents from HuggingFace dataset")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading from HuggingFace: {e}")
            logger.info("Falling back to local CSV if available...")
            return self.load_documents_from_csv(COMBINED_DATASET_PATH)
    
    def load_documents_from_csv(self, csv_path: Path) -> List[LCDocument]:
        """Load documents from the combined CSV dataset (fallback method)"""
        logger.info(f"Loading documents from {csv_path}...")
        
        if not csv_path.exists():
            raise FileNotFoundError(f"Dataset not found: {csv_path}")
        
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df):,} entries from CSV")
        
        documents = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Converting to documents"):
            metadata = {
                'source': row['source'],
                'doc_id': idx,
                'char_length': len(row['text'])
            }
            
            doc = LCDocument(
                page_content=row['text'],
                metadata=metadata
            )
            documents.append(doc)
        
        logger.info(f"Created {len(documents):,} LangChain documents")
        return documents
    
    def create_index(self, documents: List[LCDocument]):
        """Create FAISS index from documents"""
        logger.info("Creating FAISS index...")
        
        if not documents:
            raise ValueError("No documents provided for indexing")
        
        # Create vector store from documents
        logger.info("Computing embeddings and building index...")
        self.vector_store = FAISS.from_documents(
            documents=documents,
            embedding=self.embedding_function
        )
        
        logger.info(f"Index created successfully with {len(documents):,} documents")
    
    def save_index(self, index_path: Path):
        """Save the FAISS index to disk"""
        if self.vector_store is None:
            raise ValueError("No index to save. Create an index first.")
        
        logger.info(f"Saving index to {index_path}...")
        index_path.parent.mkdir(parents=True, exist_ok=True)
        self.vector_store.save_local(str(index_path))
        logger.info("Index saved successfully")
    
    def load_index(self, index_path: Path):
        """Load FAISS index from disk"""
        if not index_path.exists():
            raise FileNotFoundError(f"Index not found: {index_path}")
        
        logger.info(f"Loading index from {index_path}...")
        self.vector_store = FAISS.load_local(
            str(index_path), 
            self.embedding_function, 
            allow_dangerous_deserialization=True
        )
        logger.info("Index loaded successfully")
    
    def load_or_create_index(self):
        """Load existing index or create new one from HuggingFace dataset"""
        if FAISS_INDEX_PATH.exists() and (FAISS_INDEX_PATH / "index.faiss").exists():
            logger.info("Found existing FAISS index, loading...")
            self.load_index(FAISS_INDEX_PATH)
        else:
            logger.info("No existing index found, creating new one from HuggingFace repository...")
            documents = self.load_documents_from_huggingface()
            self.create_index(documents)
            self.save_index(FAISS_INDEX_PATH)
    
    def retrieve(self, query: str, top_k: int = TOP_K_DOCS) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for a query"""
        if self.vector_store is None:
            raise ValueError("No index loaded. Load or create an index first.")
        
        docs_with_scores = self.vector_store.similarity_search_with_score(query, k=top_k)
        
        results = []
        for doc, score in docs_with_scores:
            results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "similarity_score": 1.0 - score  # Convert distance to similarity
            })
        
        return results
    
    def generate_answer_streaming(self, query: str, context: str) -> str:
        """Generate an answer with real-time token streaming"""
        if not self._generator_loaded:
            return "Generator model not available. Only retrieval results shown."
        
        prompt = f"""Based on the following context, answer the question. Be concise and accurate.

Context: {context}

Question: {query}

Answer:"""
        
        try:
            # Tokenize the prompt
            inputs = self.generator_tokenizer.encode(prompt, return_tensors="pt").to(self.generator_model.device)
            
            # Create streamer
            streamer = TextIteratorStreamer(
                self.generator_tokenizer, 
                timeout=60.0, 
                skip_prompt=True, 
                skip_special_tokens=True
            )
            
            # Generation parameters
            generation_kwargs = {
                "input_ids": inputs,
                "max_new_tokens": OUTPUT_MAX_LENGTH,
                "temperature": 0.7,
                "do_sample": True,
                "streamer": streamer,
                "pad_token_id": self.generator_tokenizer.eos_token_id,
            }
            
            # Start generation in a separate thread
            thread = threading.Thread(target=self.generator_model.generate, kwargs=generation_kwargs)
            thread.start()
            
            # Stream and collect tokens
            print("Answer: ", end="", flush=True)
            generated_text = ""
            for new_text in streamer:
                if new_text:
                    print(new_text, end="", flush=True)
                    generated_text += new_text
            
            print()  # New line after streaming
            thread.join()  # Wait for generation to complete
            
            return generated_text.strip() if generated_text else "No answer generated."
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "Error generating answer. Please try again."
    
    def generate_answer(self, query: str, context: str) -> str:
        """Generate an answer based on query and context (non-streaming fallback)"""
        if not self._generator_loaded:
            return "Generator model not available. Only retrieval results shown."
        
        prompt = f"""Based on the following context, answer the question. Be concise and accurate.

Context: {context}

Question: {query}

Answer:"""
        
        try:
            inputs = self.generator_tokenizer.encode(prompt, return_tensors="pt").to(self.generator_model.device)
            
            with torch.no_grad():
                outputs = self.generator_model.generate(
                    inputs,
                    max_new_tokens=OUTPUT_MAX_LENGTH,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.generator_tokenizer.eos_token_id,
                )
            
            # Decode only the newly generated tokens
            generated_tokens = outputs[0][inputs.shape[1]:]
            response = self.generator_tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            return response.strip() if response else "No answer generated."
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "Error generating answer. Please try again."
    
    def answer(self, query: str, top_k: int = TOP_K_DOCS, streaming: bool = True) -> Dict[str, Any]:
        """Complete RAG pipeline: retrieve and generate"""
        logger.info(f"Processing query: {query}")
        
        # Retrieve relevant documents
        retrieved_docs = self.retrieve(query, top_k)
        
        # Prepare context
        if retrieved_docs:
            context = "\n\n".join([doc["content"] for doc in retrieved_docs])
        else:
            context = "No relevant documents found."
        
        # Generate answer (with or without streaming)
        if streaming and self._generator_loaded:
            answer = self.generate_answer_streaming(query, context)
        else:
            answer = self.generate_answer(query, context)
        
        return {
            "question": query,
            "answer": answer,
            "retrieved_documents": retrieved_docs,
            "context": context
        }

def interactive_test(rag: RAG):
    """Interactive testing interface with streaming"""
    print("\n" + "="*60)
    print("RAG SYSTEM INTERACTIVE TEST")
    print("="*60)
    print("Type your questions and press Enter.")
    print("Type 'quit', 'exit', or 'q' to stop.")
    print("-"*60)
    
    while True:
        try:
            # Get user input
            query = input("\nQuestion: ").strip()
            
            # Check for exit commands
            if query.lower() in ['quit', 'exit', 'q', '']:
                print("Goodbye!")
                break
            
            # Process the query with streaming always enabled
            print("Searching documents...")
            result = rag.answer(query, streaming=True)
            
            # Display retrieved documents info
            print(f"\nRetrieved {len(result['retrieved_documents'])} documents:")
            for i, doc in enumerate(result['retrieved_documents'], 1):
                source = doc['metadata'].get('source', 'Unknown')
                similarity = doc['similarity_score']
                content_preview = doc['content'][:100] + "..." if len(doc['content']) > 100 else doc['content']
                print(f"   {i}. Source: {source} (Score: {similarity:.3f})")
                print(f"      Preview: {content_preview}")
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            print("Please try again with a different question.")

def main():
    """Main function to run the RAG system"""
    print("Starting RAG System...")
    
    try:
        # Initialize RAG
        rag = RAG()
        
        # Load or create index
        rag.load_or_create_index()
        
        print("RAG system ready!")
        
        # Start interactive testing
        interactive_test(rag)
        
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")
        print(f"Error: {e}")
        print("Please check your data files and try again.")

if __name__ == "__main__":
    main()