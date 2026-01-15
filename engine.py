"""
ByteMe Engine - Core RAG Engine with Hybrid Search
Based on the agentic architecture from the notebook
"""

import os
import torch
import numpy as np
import warnings
from typing import List, Dict, Any
import chromadb
from PIL import Image as PILImage

warnings.filterwarnings("ignore")

# Check for GPU availability
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class ByteMeEngine:
    """
    Core RAG Engine with Hybrid Search capabilities.
    Supports both text and vision-based retrieval using ChromaDB.
    """
    
    def __init__(self, db_path: str = "./chroma_db", use_vision: bool = False):
        """
        Initialize the ByteMe Engine.
        
        Args:
            db_path: Path to ChromaDB persistent storage
            use_vision: Whether to load vision models (ColPali) - requires GPU
        """
        self.db_path = db_path
        self.use_vision = use_vision and DEVICE == "cuda"
        self.colpali_model = None
        self.colpali_processor = None
        self.dense_embedder = None
        self.bm25_retriever = None
        
        print(f"🔧 Initializing ByteMeEngine...")
        print(f"   Hardware: {DEVICE}")
        print(f"   Vision Model: {'Enabled' if self.use_vision else 'Disabled'}")
        
        self._initialize_embedder()
        self._initialize_chromadb()
        
        if self.use_vision:
            self._initialize_vision_model()
        
        print("✅ ByteMeEngine initialized successfully")
    
    def _initialize_embedder(self):
        """Initialize the text embedding model"""
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            
            self.dense_embedder = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': DEVICE}
            )
            print("   ✓ Text embedder loaded (MiniLM)")
        except Exception as e:
            print(f"   ✗ Failed to load embedder: {e}")
            raise
    
    def _initialize_chromadb(self):
        """Initialize ChromaDB collections"""
        try:
            self.client = chromadb.PersistentClient(path=self.db_path)
            
            # Text collection (384 dimensions for MiniLM)
            self.text_collection = self.client.get_or_create_collection(
                name="text_store",
                metadata={"hnsw:space": "cosine"}
            )
            
            # Vision collection (for ColPali embeddings)
            self.vision_collection = self.client.get_or_create_collection(
                name="vision_store",
                metadata={"hnsw:space": "cosine"}
            )
            
            # Long-term memory collection
            self.memory_collection = self.client.get_or_create_collection(
                name="long_term_memory",
                metadata={"hnsw:space": "cosine"}
            )
            
            print(f"   ✓ ChromaDB initialized at {self.db_path}")
            print(f"     - Text documents: {self.text_collection.count()}")
            print(f"     - Vision pages: {self.vision_collection.count()}")
            print(f"     - Memory records: {self.memory_collection.count()}")
        except Exception as e:
            print(f"   ✗ Failed to initialize ChromaDB: {e}")
            raise
    
    def _initialize_vision_model(self):
        """Initialize ColPali vision model (optional, requires GPU)"""
        try:
            from colpali_engine.models import ColPali, ColPaliProcessor
            from transformers import BitsAndBytesConfig
            
            print("   >> Loading ColPali (Vision Model)...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
            
            self.colpali_model = ColPali.from_pretrained(
                "vidore/colpali-v1.2",
                quantization_config=bnb_config,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            self.colpali_processor = ColPaliProcessor.from_pretrained("vidore/colpali-v1.2")
            print("   ✓ Vision model loaded (ColPali)")
        except Exception as e:
            print(f"   ⚠ Vision model not available: {e}")
            self.use_vision = False
    
    def process_pdf(self, pdf_path: str, domain: str = "general"):
        """
        Process a PDF document and index it into ChromaDB.
        
        Args:
            pdf_path: Path to the PDF file
            domain: Domain category (it_service, developer, hr, general)
        """
        try:
            import fitz  # PyMuPDF
            from langchain_experimental.text_splitter import SemanticChunker
        except ImportError:
            print("❌ Required packages not installed. Run: pip install pymupdf langchain-experimental")
            return
        
        print(f"🚀 Ingesting: {pdf_path}")
        doc = fitz.open(pdf_path)
        
        txt_ids, txt_vecs, txt_metas, txt_docs = [], [], [], []
        full_text_list = []
        
        # Extract text from all pages
        for page_num, page in enumerate(doc):
            page_text = page.get_text()
            full_text_list.append(page_text)
        
        # Semantic chunking
        print(">> Semantic Chunking & Text Embedding...")
        text_splitter = SemanticChunker(self.dense_embedder)
        text_chunks = text_splitter.create_documents(full_text_list)
        text_embeddings = self.dense_embedder.embed_documents([t.page_content for t in text_chunks])
        
        for i, chunk in enumerate(text_chunks):
            txt_ids.append(f"{domain}_txt_{i}")
            txt_vecs.append(text_embeddings[i])
            txt_metas.append({
                "source": pdf_path,
                "type": "text",
                "domain": domain
            })
            txt_docs.append(chunk.page_content)
        
        # Upsert to ChromaDB
        if txt_ids:
            print(f">> Indexing {len(txt_ids)} Text Chunks...")
            self.text_collection.upsert(
                ids=txt_ids,
                embeddings=txt_vecs,
                metadatas=txt_metas,
                documents=txt_docs
            )
        
        print("✅ Ingestion Complete.")
        return len(txt_ids)
    
    def hybrid_search(self, query: str, domain: str = None, k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform hybrid search across text and vision collections.
        
        Args:
            query: Search query
            domain: Optional domain filter
            k: Number of results to return
            
        Returns:
            List of search results with metadata
        """
        results = []
        
        # Text Dense Search
        q_dense = self.dense_embedder.embed_query(query)
        
        # Build where clause for domain filtering
        where_clause = {"domain": domain} if domain else None
        
        txt_res = self.text_collection.query(
            query_embeddings=[q_dense],
            n_results=k,
            where=where_clause
        )
        
        if txt_res['documents'] and txt_res['documents'][0]:
            for i, doc in enumerate(txt_res['documents'][0]):
                results.append({
                    "content": doc,
                    "metadata": txt_res['metadatas'][0][i] if txt_res['metadatas'] else {},
                    "type": "text",
                    "distance": txt_res['distances'][0][i] if txt_res['distances'] else None
                })
        
        # Vision Search (if available)
        if self.use_vision and self.colpali_model:
            with torch.no_grad():
                batch = self.colpali_processor.process_queries([query]).to(DEVICE)
                emb = self.colpali_model(**batch)
                q_vis = torch.mean(emb, dim=1).float().cpu().numpy()[0].tolist()
            
            vis_res = self.vision_collection.query(
                query_embeddings=[q_vis],
                n_results=k,
                where=where_clause
            )
            
            if vis_res['documents'] and vis_res['documents'][0]:
                for i, doc in enumerate(vis_res['documents'][0]):
                    results.append({
                        "content": doc,
                        "metadata": vis_res['metadatas'][0][i] if vis_res['metadatas'] else {},
                        "type": "vision",
                        "distance": vis_res['distances'][0][i] if vis_res['distances'] else None
                    })
        
        # Deduplicate by content
        seen = set()
        unique_results = []
        for r in results:
            if r['content'] not in seen:
                seen.add(r['content'])
                unique_results.append(r)
        
        return unique_results[:k]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics"""
        return {
            "text_documents": self.text_collection.count(),
            "vision_pages": self.vision_collection.count(),
            "memory_records": self.memory_collection.count(),
            "device": DEVICE,
            "vision_enabled": self.use_vision
        }


# Singleton instance for the app
_engine_instance = None

def get_engine(db_path: str = "./chroma_db", use_vision: bool = False) -> ByteMeEngine:
    """Get or create the ByteMe Engine singleton"""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = ByteMeEngine(db_path=db_path, use_vision=use_vision)
    return _engine_instance
