import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import json
import os
import logging
import threading
import time
from pathlib import Path

# Configure logging early
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths - consistent with other files in the project
DATA_DIR = Path(__file__).parent / "data"
FAISS_INDEX_PATH = DATA_DIR / "faiss_index"
CHUNKS_DATA_PATH = DATA_DIR / "combined_dataset_100k.csv"

class RAGTester:
    def __init__(self, master):
        self.master = master
        master.title("RAG Testing Interface")
        master.geometry("1200x800")

        # Initialize all ML-related variables as None (lazy loading)
        self.embedding_model = None
        self.generator_model = None
        self.tokenizer = None
        self.generator_pipeline = None
        self.faiss_index = None
        self.vector_store = None
        self.chunks_data = None
        self.device = None
        
        # ML libraries will be imported only when needed
        self._ml_libraries_loaded = False
        self._system_initialized = False
        
        self.logger = logger
        self.setup_ui()

    def setup_ui(self):
        # Main container
        main_frame = ttk.Frame(self.master)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # --- Top Section: Initialization ---
        init_frame = ttk.LabelFrame(main_frame, text="RAG System Initialization")
        init_frame.pack(fill=tk.X, pady=(0, 10))

        self.init_button = ttk.Button(init_frame, text="Initialize RAG System", command=self.start_initialization)
        self.init_button.pack(pady=10, padx=10)

        self.status_label = ttk.Label(init_frame, text="Click 'Initialize RAG System' to start")
        self.status_label.pack(pady=(0, 10))

        # --- Middle Section: Query Input ---
        query_frame = ttk.LabelFrame(main_frame, text="Query Input")
        query_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(query_frame, text="Enter your question:").pack(anchor='w', padx=10, pady=(10, 5))
        
        self.query_text = scrolledtext.ScrolledText(query_frame, wrap=tk.WORD, height=4)
        self.query_text.pack(fill=tk.X, padx=10, pady=5)
        self.query_text.config(state=tk.DISABLED)

        self.submit_button = ttk.Button(query_frame, text="Submit Query", command=self.start_query_processing)
        self.submit_button.pack(pady=10)
        self.submit_button.config(state=tk.DISABLED)

        # --- Bottom Section: Results ---
        results_frame = ttk.Frame(main_frame)
        results_frame.pack(fill=tk.BOTH, expand=True)

        # Notebook for tabbed results
        self.notebook = ttk.Notebook(results_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Tab 1: RAG Answer
        self.rag_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.rag_frame, text='RAG Answer')
        
        ttk.Label(self.rag_frame, text="RAG Model Answer:", font=('Helvetica', 12, 'bold')).pack(anchor='w', padx=10, pady=(10, 5))
        self.rag_answer_text = scrolledtext.ScrolledText(self.rag_frame, wrap=tk.WORD, state='disabled')
        self.rag_answer_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Tab 2: Base Model Answer
        self.base_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.base_frame, text='Base Model Answer')
        
        ttk.Label(self.base_frame, text="Base Model Answer (No RAG):", font=('Helvetica', 12, 'bold')).pack(anchor='w', padx=10, pady=(10, 5))
        self.base_answer_text = scrolledtext.ScrolledText(self.base_frame, wrap=tk.WORD, state='disabled')
        self.base_answer_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Tab 3: Retrieved Documents
        self.docs_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.docs_frame, text='Retrieved Documents')
        
        ttk.Label(self.docs_frame, text="Retrieved Documents:", font=('Helvetica', 12, 'bold')).pack(anchor='w', padx=10, pady=(10, 5))
        self.docs_text = scrolledtext.ScrolledText(self.docs_frame, wrap=tk.WORD, state='disabled')
        self.docs_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Tab 4: Complete Prompt
        self.prompt_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.prompt_frame, text='Complete Prompt')
        
        ttk.Label(self.prompt_frame, text="Complete Prompt sent to RAG Model:", font=('Helvetica', 12, 'bold')).pack(anchor='w', padx=10, pady=(10, 5))
        self.prompt_text = scrolledtext.ScrolledText(self.prompt_frame, wrap=tk.WORD, state='disabled')
        self.prompt_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Status bar
        self.status_bar = ttk.Label(self.master, text="Ready to initialize RAG system", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def update_status(self, message):
        """Thread-safe status update"""
        def _update():
            self.status_label.config(text=message)
            self.status_bar.config(text=message)
        
        if threading.current_thread() == threading.main_thread():
            _update()
        else:
            self.master.after(0, _update)

    def load_ml_libraries(self):
        """Lazy loading of ML libraries to prevent startup crashes"""
        if self._ml_libraries_loaded:
            return True
            
        try:
            self.update_status("Loading ML libraries...")
            
            # Import ML libraries only when needed
            global torch, faiss, np, pd, SentenceTransformer, AutoTokenizer, AutoModelForCausalLM, hf_pipeline, gc
            
            import torch
            import faiss
            import numpy as np
            import pandas as pd
            from sentence_transformers import SentenceTransformer
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline as hf_pipeline
            import gc
            
            # Device setup with better detection
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                self.logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
                self.logger.info("Using Apple Silicon GPU (MPS)")
            else:
                self.device = torch.device("cpu")
                self.logger.info("Using CPU")
            
            self._ml_libraries_loaded = True
            self.logger.info("ML libraries loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load ML libraries: {e}")
            return False

    def cleanup_memory(self):
        """Clean up GPU/MPS memory"""
        if not self._ml_libraries_loaded or self.device is None:
            return
            
        try:
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            elif self.device.type == "mps":
                torch.mps.empty_cache()
            gc.collect()
        except Exception as e:
            self.logger.warning(f"Memory cleanup failed: {e}")

    def load_generator_model(self):
        """Load generator model with fallback options and better error handling"""
        # List of models to try in order of preference
        model_candidates = [
            "Lysandrec/MNLP_M3_rag_model",
            "HAissa/MNLP_M3_mcqa_model",
        ]
        
        for model_name in model_candidates:
            try:
                self.logger.info(f"Attempting to load model: {model_name}")
                self.update_status(f"Loading generator model: {model_name}...")
                
                # Clear memory before loading
                self.cleanup_memory()
                
                # Load tokenizer first
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name, 
                    trust_remote_code=True,
                    padding_side='left'
                )
                
                # Add padding token if it doesn't exist
                if tokenizer.pad_token is None:
                    if tokenizer.eos_token is not None:
                        tokenizer.pad_token = tokenizer.eos_token
                    else:
                        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                
                # Load model with appropriate dtype
                model_kwargs = {
                    "trust_remote_code": True,
                    "torch_dtype": torch.float16 if self.device.type != "cpu" else torch.float32,
                    "low_cpu_mem_usage": True
                }
                
                # Try loading model
                model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
                
                # Resize embeddings if needed
                if len(tokenizer) != model.config.vocab_size:
                    model.resize_token_embeddings(len(tokenizer))
                
                # Move to device
                model = model.to(self.device)
                
                # Create pipeline
                pipeline = hf_pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=200,
                    temperature=0.7,
                    do_sample=True,
                    device=self.device if self.device.type != "mps" else -1,  # MPS compatibility
                    pad_token_id=tokenizer.pad_token_id
                )
                
                # Success! Store the loaded components
                self.tokenizer = tokenizer
                self.generator_model = model
                self.generator_pipeline = pipeline
                
                self.logger.info(f"Successfully loaded model: {model_name}")
                return True
                
            except Exception as e:
                self.logger.warning(f"Failed to load {model_name}: {str(e)}")
                # Clean up any partially loaded components
                self.cleanup_memory()
                continue
        
        # If we get here, all models failed
        raise Exception("Failed to load any generator model. Please check your environment and model availability.")

    def start_initialization(self):
        """Start initialization in a separate thread to prevent GUI blocking"""
        self.init_button.config(state=tk.DISABLED)
        self.update_status("Starting initialization...")
        
        def initialization_thread():
            try:
                self.initialize_rag_system()
            except Exception as e:
                error_msg = str(e)  # Capture the error message
                self.master.after(0, lambda msg=error_msg: self.handle_initialization_error(msg))
        
        thread = threading.Thread(target=initialization_thread, daemon=True)
        thread.start()

    def handle_initialization_error(self, error_msg):
        """Handle initialization errors on the main thread"""
        full_error = f"Failed to initialize RAG system: {error_msg}"
        self.logger.error(full_error)
        self.update_status(full_error)
        messagebox.showerror("Initialization Error", full_error)
        self.init_button.config(state=tk.NORMAL)
        self.cleanup_memory()

    def handle_initialization_success(self):
        """Handle successful initialization on the main thread"""
        self.query_text.config(state=tk.NORMAL)
        self.submit_button.config(state=tk.NORMAL)
        self.update_status("RAG system initialized successfully! Ready for queries.")
        messagebox.showinfo("Success", "RAG system initialized successfully!")

    def initialize_rag_system(self):
        try:
            self.update_status("Initializing RAG system... This may take a few minutes.")

            # Load ML libraries first
            if not self.load_ml_libraries():
                raise Exception("Failed to load ML libraries")

            # Load embedding model
            self.update_status("Loading embedding model...")
            self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            self.logger.info("Embedding model loaded successfully")

            # Load generator model with fallback
            self.load_generator_model()

            # Load FAISS index using LangChain format
            self.update_status("Loading FAISS index...")
            if not FAISS_INDEX_PATH.exists():
                raise FileNotFoundError(f"FAISS index not found at {FAISS_INDEX_PATH}")
            
            # Check if it's a LangChain FAISS directory format
            if FAISS_INDEX_PATH.is_dir() and (FAISS_INDEX_PATH / "index.faiss").exists():
                # LangChain FAISS format - need to use langchain's FAISS loader
                from langchain_community.vectorstores import FAISS
                from langchain_huggingface import HuggingFaceEmbeddings
                
                # Create embedding function compatible with LangChain
                embedding_function = HuggingFaceEmbeddings(
                    model_name='sentence-transformers/all-MiniLM-L6-v2'
                )
                
                # Load using LangChain FAISS
                self.vector_store = FAISS.load_local(
                    str(FAISS_INDEX_PATH), 
                    embedding_function, 
                    allow_dangerous_deserialization=True
                )
                self.logger.info("FAISS index loaded successfully using LangChain format")
            else:
                # Single file format - use raw FAISS
                self.faiss_index = faiss.read_index(str(FAISS_INDEX_PATH))
                self.logger.info("FAISS index loaded successfully using raw format")

            # Load chunks data
            self.update_status("Loading chunks data...")
            if not CHUNKS_DATA_PATH.exists():
                raise FileNotFoundError(f"Chunks data not found at {CHUNKS_DATA_PATH}")
            
            self.chunks_data = pd.read_csv(CHUNKS_DATA_PATH)
            self.logger.info(f"Loaded {len(self.chunks_data)} chunks")

            # Final memory cleanup
            self.cleanup_memory()

            # Mark as initialized
            self._system_initialized = True

            # Schedule UI update on main thread
            self.master.after(0, self.handle_initialization_success)

        except Exception as e:
            # Re-raise for thread handler
            raise e

    def retrieve_documents(self, query, top_k=5):
        """Retrieve top-k most similar documents for the query"""
        if hasattr(self, 'vector_store') and self.vector_store is not None:
            # Use LangChain FAISS vector store
            docs_with_scores = self.vector_store.similarity_search_with_score(query, k=top_k)
            
            retrieved_docs = []
            for i, (doc, score) in enumerate(docs_with_scores):
                retrieved_docs.append({
                    'rank': i + 1,
                    'similarity': 1.0 - score,  # Convert distance to similarity
                    'text': doc.page_content
                })
            
            return retrieved_docs
        else:
            # Use raw FAISS index with sentence transformer
            query_embedding = self.embedding_model.encode([query])
            
            # Search in FAISS index
            similarities, indices = self.faiss_index.search(query_embedding.astype('float32'), top_k)
            
            retrieved_docs = []
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if idx < len(self.chunks_data):
                    doc_text = self.chunks_data.iloc[idx]['text']
                    retrieved_docs.append({
                        'rank': i + 1,
                        'similarity': float(similarity),
                        'text': doc_text
                    })
            
            return retrieved_docs

    def generate_answer(self, query, retrieved_docs=None, max_length=200):
        """Generate answer using the generator pipeline with better error handling"""
        try:
            if retrieved_docs:
                # Create context from retrieved documents
                context = "\n\n".join([doc['text'][:500] for doc in retrieved_docs[:3]])  # Use top 3 docs
                prompt = f"Based on the following context, answer the question. If the context doesn't contain the answer, say so.\n\nContext: {context}\n\nQuestion: {query}\n\nAnswer:"
            else:
                prompt = f"Question: {query}\nAnswer:"

            # Use the pipeline for generation
            response = self.generator_pipeline(
                prompt,
                max_new_tokens=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                truncation=True
            )
            
            # Extract the generated text
            generated_text = response[0]['generated_text']
            
            # Extract only the answer part
            if "Answer:" in generated_text:
                answer = generated_text.split("Answer:")[-1].strip()
            else:
                answer = generated_text[len(prompt):].strip()
            
            # Clean up the answer
            answer = answer.split('\n')[0].strip()  # Take only the first line
            
            return {
                'answer': answer if answer else "I couldn't generate a clear answer for this question.",
                'prompt': prompt
            }
            
        except Exception as e:
            self.logger.error(f"Error in text generation: {e}")
            return {
                'answer': f"Error generating answer: {str(e)}",
                'prompt': f"Error occurred while generating prompt: {str(e)}"
            }

    def start_query_processing(self):
        """Start query processing in a separate thread"""
        if not self._system_initialized:
            messagebox.showwarning("System Not Ready", "Please initialize the RAG system first.")
            return
            
        query = self.query_text.get('1.0', tk.END).strip()
        if not query:
            messagebox.showwarning("Input Error", "Please enter a question.")
            return

        self.submit_button.config(state=tk.DISABLED)
        self.update_status("Processing query...")
        
        def query_thread():
            try:
                self.process_query(query)
            except Exception as e:
                error_msg = str(e)  # Capture the error message
                self.master.after(0, lambda msg=error_msg: self.handle_query_error(msg))
            finally:
                self.master.after(0, lambda: self.submit_button.config(state=tk.NORMAL))
        
        thread = threading.Thread(target=query_thread, daemon=True)
        thread.start()

    def handle_query_error(self, error_msg):
        """Handle query errors on the main thread"""
        full_error = f"Error processing query: {error_msg}"
        self.logger.error(full_error)
        self.update_status(full_error)
        messagebox.showerror("Query Error", full_error)

    def process_query(self, query):
        """Process the query (runs in background thread)"""
        try:
            # Retrieve documents
            retrieved_docs = self.retrieve_documents(query, top_k=5)
            
            # Generate RAG answer
            rag_result = self.generate_answer(query, retrieved_docs)
            rag_answer = rag_result['answer']
            rag_prompt = rag_result['prompt']
            
            # Generate base model answer (without RAG)
            base_result = self.generate_answer(query, retrieved_docs=None)
            base_answer = base_result['answer']

            # Prepare docs text
            docs_text = ""
            for doc in retrieved_docs:
                docs_text += f"--- Document {doc['rank']} (Similarity: {doc['similarity']:.4f}) ---\n"
                docs_text += f"{doc['text']}\n\n"
            
            if not docs_text:
                docs_text = "No documents retrieved."

            # Prepare complete prompt display
            prompt_display = self.format_prompt_display(query, retrieved_docs, rag_prompt)

            # Schedule UI update on main thread
            def update_results():
                self.display_text(self.rag_answer_text, rag_answer)
                self.display_text(self.base_answer_text, base_answer)
                self.display_text(self.docs_text, docs_text)
                self.display_text(self.prompt_text, prompt_display)
                self.update_status(f"Query processed successfully. Retrieved {len(retrieved_docs)} documents.")

            self.master.after(0, update_results)

            # Clean up memory after processing
            self.cleanup_memory()

        except Exception as e:
            # Re-raise for thread handler
            raise e

    def format_prompt_display(self, query, retrieved_docs, full_prompt):
        """Format the prompt display with clear sections"""
        display = "=" * 80 + "\n"
        display += "COMPLETE RAG PROMPT BREAKDOWN\n"
        display += "=" * 80 + "\n\n"
        
        # Original query section
        display += "ORIGINAL QUERY:\n"
        display += "-" * 40 + "\n"
        display += f"{query}\n\n"
        
        # Retrieved documents section
        display += f"RETRIEVED DOCUMENTS ({len(retrieved_docs)} documents):\n"
        display += "-" * 40 + "\n"
        for i, doc in enumerate(retrieved_docs[:3], 1):  # Show top 3 docs used in context
            display += f"Document {i} (Similarity: {doc['similarity']:.4f}):\n"
            display += f"{doc['text'][:300]}{'...' if len(doc['text']) > 300 else ''}\n\n"
        
        # Complete prompt section
        display += "COMPLETE PROMPT SENT TO MODEL:\n"
        display += "-" * 40 + "\n"
        display += full_prompt + "\n\n"
        
        return display

    def display_text(self, widget, text):
        """Helper method to display text in a text widget"""
        widget.config(state='normal')
        widget.delete('1.0', tk.END)
        widget.insert(tk.END, text)
        widget.config(state='disabled')

if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = RAGTester(root)
        root.mainloop()
    except Exception as e:
        print(f"Failed to start GUI: {e}")
        import traceback
        traceback.print_exc()