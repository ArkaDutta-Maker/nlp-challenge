"""
Memory Manager - Short-term (Redis) and Long-term (PostgreSQL/ChromaDB) Memory
Based on the agentic architecture from the notebook
Supports PostgreSQL (Neon DB) for persistent user-specific memory storage
"""

import json
import hashlib
from datetime import datetime
from collections import deque
from typing import List, Dict, Any, Optional
import os

# Redis support (optional - falls back to in-memory if not available)
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("⚠️ Redis not installed. Using in-memory short-term storage.")

# PostgreSQL support for long-term memory
try:
    from database import get_database, DatabaseManager
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
    print("⚠️ PostgreSQL database module not available.")


class ShortTermMemory:
    """
    Short-term memory implementation using Redis or in-memory fallback.
    Stores recent conversation exchanges per session.
    """
    
    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        redis_password: str = None,
        max_exchanges: int = 10,
        ttl_seconds: int = 3600  # 1 hour default TTL
    ):
        self.max_exchanges = max_exchanges
        self.ttl_seconds = ttl_seconds
        self.redis_client = None
        self.use_redis = False
        
        # Try to connect to Redis
        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.Redis(
                    host=redis_host,
                    port=redis_port,
                    db=redis_db,
                    password=redis_password,
                    decode_responses=True,
                    socket_connect_timeout=2
                )
                # Test connection
                self.redis_client.ping()
                self.use_redis = True
                print(f"✅ Redis connected at {redis_host}:{redis_port}")
            except Exception as e:
                print(f"⚠️ Redis connection failed: {e}. Using in-memory storage.")
                self.redis_client = None
        
        # Fallback: In-memory storage
        self._memory_store: Dict[str, deque] = {}
    
    def _get_key(self, session_id: str, user_id: str) -> str:
        """Generate Redis key for session"""
        return f"byteme:session:{user_id}:{session_id}"
    
    def add_exchange(
        self,
        session_id: str,
        user_id: str,
        question: str,
        answer: str,
        metadata: Dict = None
    ) -> Dict:
        """Add a Q&A exchange to short-term memory"""
        exchange = {
            "question": question,
            "answer": answer,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        if self.use_redis and self.redis_client:
            key = self._get_key(session_id, user_id)
            # Get existing exchanges
            data = self.redis_client.get(key)
            exchanges = json.loads(data) if data else []
            
            # Add new exchange (FIFO)
            exchanges.append(exchange)
            if len(exchanges) > self.max_exchanges:
                exchanges = exchanges[-self.max_exchanges:]
            
            # Store back with TTL
            self.redis_client.setex(key, self.ttl_seconds, json.dumps(exchanges))
        else:
            # In-memory fallback
            key = self._get_key(session_id, user_id)
            if key not in self._memory_store:
                self._memory_store[key] = deque(maxlen=self.max_exchanges)
            self._memory_store[key].append(exchange)
        
        return exchange
    
    def get_history(
        self,
        session_id: str,
        user_id: str,
        n: int = None
    ) -> List[Dict]:
        """Retrieve conversation history for a session"""
        n = n or self.max_exchanges
        
        if self.use_redis and self.redis_client:
            key = self._get_key(session_id, user_id)
            data = self.redis_client.get(key)
            if data:
                exchanges = json.loads(data)
                return exchanges[-n:]
            return []
        else:
            key = self._get_key(session_id, user_id)
            if key in self._memory_store:
                return list(self._memory_store[key])[-n:]
            return []
    
    def format_for_prompt(
        self,
        session_id: str,
        user_id: str,
        n: int = 3
    ) -> str:
        """Format recent history as a string for LLM context"""
        history = self.get_history(session_id, user_id, n)
        if not history:
            return "No previous conversation."
        
        formatted = []
        for i, exch in enumerate(history, 1):
            q = exch['question'][:100] + "..." if len(exch['question']) > 100 else exch['question']
            a = exch['answer'][:150] + "..." if len(exch['answer']) > 150 else exch['answer']
            formatted.append(f"[{i}] Q: {q}\n    A: {a}")
        
        return "\n".join(formatted)
    
    def clear_session(self, session_id: str, user_id: str):
        """Clear short-term memory for a session"""
        if self.use_redis and self.redis_client:
            key = self._get_key(session_id, user_id)
            self.redis_client.delete(key)
        else:
            key = self._get_key(session_id, user_id)
            if key in self._memory_store:
                del self._memory_store[key]
    
    def get_session_count(self, session_id: str, user_id: str) -> int:
        """Get number of exchanges in a session"""
        history = self.get_history(session_id, user_id)
        return len(history)


class LongTermMemory:
    """
    Long-term memory implementation using ChromaDB.
    Stores important conversations and facts persistently.
    """
    
    def __init__(self, chromadb_client, embedder):
        """
        Initialize long-term memory.
        
        Args:
            chromadb_client: ChromaDB client instance
            embedder: Embedding model (HuggingFace)
        """
        self.embedder = embedder
        
        # Conversation memory collection
        self.memory_collection = chromadb_client.get_or_create_collection(
            name="long_term_memory",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Facts/knowledge collection
        self.facts_collection = chromadb_client.get_or_create_collection(
            name="extracted_facts",
            metadata={"hnsw:space": "cosine"}
        )
        
        print(f"✅ Long-term memory initialized")
        print(f"   - Stored conversations: {self.memory_collection.count()}")
        print(f"   - Stored facts: {self.facts_collection.count()}")
    
    def store_conversation(
        self,
        user_id: str,
        question: str,
        answer: str,
        domain: str = "general",
        importance_score: float = 0.5
    ) -> str:
        """
        Store important conversation to long-term memory.
        
        Args:
            user_id: User identifier
            question: User question
            answer: Agent answer
            domain: Domain category
            importance_score: 0-1 importance rating
            
        Returns:
            Memory ID
        """
        content = f"Question: {question} Answer: {answer}"
        mem_id = hashlib.md5(f"{user_id}|{content}|{datetime.now().isoformat()}".encode()).hexdigest()
        
        # Create embedding
        embedding = self.embedder.embed_query(content)
        
        # Store with metadata
        self.memory_collection.upsert(
            ids=[mem_id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[{
                "user_id": user_id,
                "question": question[:500],
                "answer": answer[:1000],
                "domain": domain,
                "importance": importance_score,
                "timestamp": datetime.now().isoformat()
            }]
        )
        
        return mem_id
    
    def retrieve_relevant(
        self,
        query: str,
        user_id: str = None,
        domain: str = None,
        n_results: int = 3
    ) -> List[Dict]:
        """
        Semantic search over long-term memory.
        
        Args:
            query: Search query
            user_id: Optional user filter
            domain: Optional domain filter
            n_results: Number of results
            
        Returns:
            List of relevant memories
        """
        query_embedding = self.embedder.embed_query(query)
        
        # Build where clause
        where_clause = {}
        if user_id:
            where_clause["user_id"] = user_id
        if domain:
            where_clause["domain"] = domain
        
        results = self.memory_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_clause if where_clause else None
        )
        
        memories = []
        if results['metadatas'] and results['metadatas'][0]:
            for i, meta in enumerate(results['metadatas'][0]):
                memories.append({
                    "question": meta.get("question", ""),
                    "answer": meta.get("answer", ""),
                    "domain": meta.get("domain", ""),
                    "timestamp": meta.get("timestamp", ""),
                    "distance": results['distances'][0][i] if results['distances'] else None
                })
        
        return memories
    
    def format_for_prompt(self, memories: List[Dict]) -> str:
        """Format retrieved memories for LLM context"""
        if not memories:
            return "No relevant past conversations found."
        
        formatted = []
        for i, mem in enumerate(memories, 1):
            formatted.append(
                f"[Past {i}] Q: {mem['question'][:100]}...\n"
                f"         A: {mem['answer'][:150]}..."
            )
        
        return "\n".join(formatted)
    
    def store_fact(self, fact: str, source: str, domain: str = "general"):
        """Store extracted fact"""
        fact_id = hashlib.md5(f"{fact}{datetime.now().isoformat()}".encode()).hexdigest()
        embedding = self.embedder.embed_query(fact)
        
        self.facts_collection.upsert(
            ids=[fact_id],
            embeddings=[embedding],
            documents=[fact],
            metadatas=[{
                "source": source,
                "domain": domain,
                "timestamp": datetime.now().isoformat()
            }]
        )
    
    def get_stats(self) -> Dict[str, int]:
        """Get memory statistics"""
        return {
            "conversations": self.memory_collection.count(),
            "facts": self.facts_collection.count()
        }


class PostgresLongTermMemory:
    """
    Long-term memory implementation using PostgreSQL (Neon DB).
    Stores conversations per user with embeddings for semantic search.
    """
    
    def __init__(self, db_manager: 'DatabaseManager', embedder):
        """
        Initialize PostgreSQL-based long-term memory.
        
        Args:
            db_manager: DatabaseManager instance
            embedder: Embedding model for semantic search
        """
        self.db = db_manager
        self.embedder = embedder
        
        if self.db.is_connected():
            print("✅ PostgreSQL Long-term memory initialized")
        else:
            print("⚠️ PostgreSQL not connected, using fallback")
    
    def store_conversation(
        self,
        user_id: str,
        question: str,
        answer: str,
        domain: str = "general",
        importance_score: float = 0.5
    ) -> Optional[str]:
        """Store conversation to PostgreSQL"""
        if not self.db.is_connected():
            return None
        
        # Create embedding for semantic search
        content = f"Question: {question} Answer: {answer}"
        embedding = self.embedder.embed_query(content)
        
        return self.db.store_memory(
            user_id=user_id,
            question=question,
            answer=answer,
            domain=domain,
            embedding=embedding,
            importance_score=importance_score
        )
    
    def retrieve_relevant(
        self,
        query: str,
        user_id: str = None,
        domain: str = None,
        n_results: int = 3
    ) -> List[Dict]:
        """
        Retrieve relevant memories using text search.
        For full semantic search, embeddings are stored in PostgreSQL.
        """
        if not self.db.is_connected():
            return []
        
        # Use text-based search from PostgreSQL
        memories = self.db.search_memories_by_text(
            user_id=user_id,
            search_text=query,
            domain=domain,
            limit=n_results
        )
        
        return memories
    
    def format_for_prompt(self, memories: List[Dict]) -> str:
        """Format retrieved memories for LLM context"""
        if not memories:
            return "No relevant past conversations found."
        
        formatted = []
        for i, mem in enumerate(memories, 1):
            q = mem.get('question', '')[:100]
            a = mem.get('answer', '')[:150]
            formatted.append(f"[Past {i}] Q: {q}...\n         A: {a}...")
        
        return "\n".join(formatted)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        return {"backend": "postgresql", "connected": self.db.is_connected()}


class MemoryManager:
    """
    Unified Memory Manager combining short-term (Redis) and long-term (PostgreSQL/ChromaDB) memory.
    """
    
    def __init__(
        self,
        chromadb_client=None,
        embedder=None,
        db_manager: 'DatabaseManager' = None,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_password: str = None,
        max_short_term: int = 10,
        use_postgres_memory: bool = True
    ):
        """
        Initialize unified memory manager.
        
        Args:
            chromadb_client: ChromaDB client for long-term storage (fallback)
            embedder: Embedding model
            db_manager: PostgreSQL DatabaseManager for long-term storage (preferred)
            redis_host: Redis host for short-term storage
            redis_port: Redis port
            redis_password: Redis password (optional)
            max_short_term: Max exchanges in short-term memory
            use_postgres_memory: Whether to prefer PostgreSQL for long-term memory
        """
        self.embedder = embedder
        self.db_manager = db_manager
        
        # Short-term memory (Redis or in-memory)
        self.short_term = ShortTermMemory(
            redis_host=redis_host,
            redis_port=redis_port,
            redis_password=redis_password,
            max_exchanges=max_short_term
        )
        
        # Long-term memory (PostgreSQL preferred, ChromaDB fallback)
        self.use_postgres = use_postgres_memory and db_manager and db_manager.is_connected()
        
        if self.use_postgres:
            self.long_term = PostgresLongTermMemory(db_manager, embedder)
            print("   📦 Long-term storage: PostgreSQL")
        elif chromadb_client and embedder:
            self.long_term = LongTermMemory(chromadb_client, embedder)
            print("   📦 Long-term storage: ChromaDB")
        else:
            self.long_term = None
            print("   ⚠️ Long-term storage: Not available")
        
        print("✅ Memory Manager initialized")
    
    def add_exchange(
        self,
        session_id: str,
        user_id: str,
        question: str,
        answer: str,
        domain: str = "general",
        store_long_term: bool = False,
        importance: float = 0.5
    ):
        """
        Add exchange to memory.
        
        Args:
            session_id: Current session ID
            user_id: User ID
            question: User question
            answer: Agent answer
            domain: Domain category
            store_long_term: Whether to persist to long-term
            importance: Importance score for long-term storage
        """
        # Always add to short-term
        self.short_term.add_exchange(
            session_id=session_id,
            user_id=user_id,
            question=question,
            answer=answer,
            metadata={"domain": domain}
        )
        
        # Optionally add to long-term
        if store_long_term and self.long_term:
            self.long_term.store_conversation(
                user_id=user_id,
                question=question,
                answer=answer,
                domain=domain,
                importance_score=importance
            )
            
            # Also track session in PostgreSQL if available
            if self.use_postgres and self.db_manager:
                self.db_manager.update_session_activity(session_id, user_id)
    
    def get_context(
        self,
        session_id: str,
        user_id: str,
        query: str,
        domain: str = None,
        short_term_n: int = 3,
        long_term_n: int = 2
    ) -> Dict[str, str]:
        """
        Get combined memory context for LLM.
        
        Returns:
            Dict with 'short_term' and 'long_term' context strings
        """
        # Short-term context (recent conversation)
        short_context = self.short_term.format_for_prompt(
            session_id=session_id,
            user_id=user_id,
            n=short_term_n
        )
        
        # Long-term context (relevant past conversations)
        if self.long_term:
            relevant_memories = self.long_term.retrieve_relevant(
                query=query,
                user_id=user_id,
                domain=domain,
                n_results=long_term_n
            )
            long_context = self.long_term.format_for_prompt(relevant_memories)
        else:
            long_context = "Long-term memory not available."
        
        return {
            "short_term": short_context,
            "long_term": long_context
        }
    
    def clear_session(self, session_id: str, user_id: str):
        """Clear short-term memory for a session"""
        self.short_term.clear_session(session_id, user_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get combined memory statistics"""
        stats = {
            "short_term": {
                "backend": "redis" if self.short_term.use_redis else "in-memory"
            }
        }
        
        if self.long_term:
            stats["long_term"] = self.long_term.get_stats()
        else:
            stats["long_term"] = {"backend": "none"}
        
        return stats
    
    def get_user_memory_stats(self, user_id: str) -> Dict[str, Any]:
        """Get memory statistics for a specific user (PostgreSQL only)"""
        if self.use_postgres and self.db_manager:
            return self.db_manager.get_memory_stats(user_id)
        return {"total": 0, "by_domain": {}}
    
    def clear_user_memories(self, user_id: str, domain: str = None) -> bool:
        """Clear long-term memories for a user"""
        if self.use_postgres and self.db_manager:
            return self.db_manager.clear_user_memories(user_id, domain)
        return False


# Singleton instance
_memory_manager_instance = None

def get_memory_manager(
    chromadb_client=None,
    embedder=None,
    db_manager: 'DatabaseManager' = None,
    redis_host: str = "localhost",
    redis_port: int = 6379,
    redis_password: str = None,
    use_postgres_memory: bool = True
) -> MemoryManager:
    """Get or create Memory Manager singleton"""
    global _memory_manager_instance
    if _memory_manager_instance is None:
        _memory_manager_instance = MemoryManager(
            chromadb_client=chromadb_client,
            embedder=embedder,
            db_manager=db_manager,
            redis_host=redis_host,
            redis_port=redis_port,
            redis_password=redis_password,
            use_postgres_memory=use_postgres_memory
        )
    return _memory_manager_instance
