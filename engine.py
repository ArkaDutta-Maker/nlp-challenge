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
            domain: Optional domain filter (ignored if documents don't have domain metadata)
            k: Number of results to return
            
        Returns:
            List of search results with metadata
        """
        results = []
        
        # Safety check
        if not self.dense_embedder:
            print("⚠️ Dense embedder not initialized")
            return results
        
        try:
            # Text Dense Search
            q_dense = self.dense_embedder.embed_query(query)
            
            # Query without domain filter (Kaggle-created DB doesn't have domain field)
            txt_res = self.text_collection.query(
                query_embeddings=[q_dense],
                n_results=k * 2
            )
        
            if txt_res and txt_res.get('documents') and txt_res['documents'][0]:
                for i, doc in enumerate(txt_res['documents'][0]):
                    meta = txt_res['metadatas'][0][i] if txt_res.get('metadatas') else {}
                    
                    # Extract source - handle Kaggle format (just filename) 
                    source = meta.get('source', 'Unknown Document')
                    doc_type = meta.get('type', 'text')
                    page_num = meta.get('page', None)
                    
                    # Build display source with page number if available
                    display_source = source
                    if page_num:
                        display_source = f"{source} (Page {page_num})"
                    
                    results.append({
                        "content": doc,
                        "metadata": meta,
                        "type": doc_type,
                        "distance": txt_res['distances'][0][i] if txt_res.get('distances') else None,
                        "source": display_source
                    })
        except Exception as e:
            print(f"⚠️ Text search error: {e}")
        
        # Vision Search (if available)
        if self.use_vision and self.colpali_model:
            try:
                with torch.no_grad():
                    batch = self.colpali_processor.process_queries([query]).to(DEVICE)
                    emb = self.colpali_model(**batch)
                    q_vis = torch.mean(emb, dim=1).float().cpu().numpy()[0].tolist()
                
                vis_res = self.vision_collection.query(
                    query_embeddings=[q_vis],
                    n_results=k * 2
                )
                
                if vis_res and vis_res.get('documents') and vis_res['documents'][0]:
                    for i, doc in enumerate(vis_res['documents'][0]):
                        meta = vis_res['metadatas'][0][i] if vis_res.get('metadatas') else {}
                        
                        # Extract source - handle Kaggle format
                        source = meta.get('source', 'Vision Document')
                        doc_type = meta.get('type', 'vision')
                        page_num = meta.get('page', None)
                        
                        # Build display source with page number
                        display_source = source
                        if page_num:
                            display_source = f"{source} (Page {page_num})"
                        
                        results.append({
                            "content": doc,
                            "metadata": meta,
                            "type": doc_type,
                            "distance": vis_res['distances'][0][i] if vis_res.get('distances') else None,
                            "source": display_source
                        })
            except Exception as e:
                print(f"⚠️ Vision search error: {e}")
        
        # Deduplicate by content and sort by distance
        seen = set()
        unique_results = []
        for r in results:
            content_key = r['content'][:100]  # Use first 100 chars for dedup
            if content_key not in seen:
                seen.add(content_key)
                unique_results.append(r)
        
        # Sort by distance (lower is better)
        unique_results.sort(key=lambda x: x.get('distance', float('inf')))
        return unique_results[:k]
    
    def search_by_page(self, query: str, page_number: int, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for content from a specific page number.
        
        Args:
            query: Search query (optional - if empty, returns all from page)
            page_number: The page number to search in
            k: Number of results to return
            
        Returns:
            List of search results from the specified page
        """
        results = []
        
        try:
            # Search vision collection which has page metadata
            if self.vision_collection.count() > 0:
                # Get all documents from the vision collection with page filter
                vis_res = self.vision_collection.get(
                    where={"page": page_number},
                    include=["documents", "metadatas"]
                )
                
                if vis_res and vis_res.get('documents'):
                    for i, doc in enumerate(vis_res['documents']):
                        meta = vis_res['metadatas'][i] if vis_res.get('metadatas') else {}
                        source = meta.get('source', 'Document')
                        results.append({
                            "content": doc,
                            "metadata": meta,
                            "type": "vision",
                            "source": f"{source} (Page {page_number})",
                            "page": page_number
                        })
            
            # Also search text collection (may not have page metadata)
            # If query provided, do semantic search and filter by relevance
            if query and self.dense_embedder:
                q_dense = self.dense_embedder.embed_query(query)
                txt_res = self.text_collection.query(
                    query_embeddings=[q_dense],
                    n_results=k * 3  # Get more to filter
                )
                
                if txt_res and txt_res.get('documents') and txt_res['documents'][0]:
                    for i, doc in enumerate(txt_res['documents'][0]):
                        meta = txt_res['metadatas'][0][i] if txt_res.get('metadatas') else {}
                        # Check if this doc has page metadata matching
                        doc_page = meta.get('page')
                        if doc_page == page_number:
                            results.append({
                                "content": doc,
                                "metadata": meta,
                                "type": "text",
                                "source": f"{meta.get('source', 'Document')} (Page {page_number})",
                                "page": page_number
                            })
        except Exception as e:
            print(f"⚠️ Page search error: {e}")
        
        return results[:k]
    
    def get_page_content(self, page_number: int) -> List[Dict[str, Any]]:
        """
        Get all content from a specific page.
        
        Args:
            page_number: The page number to retrieve
            
        Returns:
            List of all content from that page
        """
        results = []
        
        try:
            # Get from vision collection (has page metadata)
            if self.vision_collection.count() > 0:
                vis_res = self.vision_collection.get(
                    where={"page": page_number},
                    include=["documents", "metadatas"]
                )
                
                if vis_res and vis_res.get('documents'):
                    for i, doc in enumerate(vis_res['documents']):
                        meta = vis_res['metadatas'][i] if vis_res.get('metadatas') else {}
                        results.append({
                            "content": doc,
                            "metadata": meta,
                            "type": meta.get('type', 'vision'),
                            "source": meta.get('source', 'Document'),
                            "page": page_number
                        })
        except Exception as e:
            print(f"⚠️ Get page content error: {e}")
        
        return results
    
    def get_available_pages(self) -> List[int]:
        """Get list of available page numbers in the database"""
        pages = set()
        try:
            if self.vision_collection.count() > 0:
                # Get all metadatas
                all_data = self.vision_collection.get(include=["metadatas"])
                if all_data and all_data.get('metadatas'):
                    for meta in all_data['metadatas']:
                        if meta and 'page' in meta:
                            pages.add(meta['page'])
        except Exception as e:
            print(f"⚠️ Error getting pages: {e}")
        return sorted(list(pages))

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics"""
        available_pages = self.get_available_pages()
        return {
            "text_documents": self.text_collection.count(),
            "vision_pages": self.vision_collection.count(),
            "memory_records": self.memory_collection.count(),
            "device": DEVICE,
            "vision_enabled": self.use_vision,
            "available_pages": available_pages,
            "total_pages": len(available_pages)
        }
    
    def debug_search(self, query: str, k: int = 3) -> Dict[str, Any]:
        """Debug method to check what's in the collections"""
        debug_info = {
            "text_count": self.text_collection.count(),
            "vision_count": self.vision_collection.count(),
            "text_results": [],
            "vision_results": []
        }
        
        try:
            # Text search
            q_dense = self.dense_embedder.embed_query(query)
            txt_res = self.text_collection.query(
                query_embeddings=[q_dense],
                n_results=k
            )
            
            if txt_res and txt_res.get('documents') and txt_res['documents'][0]:
                for i, doc in enumerate(txt_res['documents'][0]):
                    meta = txt_res['metadatas'][0][i] if txt_res.get('metadatas') else {}
                    debug_info["text_results"].append({
                        "preview": doc[:150] + "..." if len(doc) > 150 else doc,
                        "metadata": meta,
                        "distance": round(txt_res['distances'][0][i], 4) if txt_res.get('distances') else None
                    })
            
            # Vision search (if collection has data)
            if self.vision_collection.count() > 0:
                # For vision, we need ColPali embeddings, but if not available, skip
                debug_info["vision_note"] = "Vision collection has data but ColPali not loaded locally"
                
        except Exception as e:
            debug_info["error"] = str(e)
        
        return debug_info


# Singleton instance for the app
_engine_instance = None

def get_engine(db_path: str = "./chroma_db", use_vision: bool = False) -> ByteMeEngine:
    """Get or create the ByteMe Engine singleton"""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = ByteMeEngine(db_path=db_path, use_vision=use_vision)
    return _engine_instance
