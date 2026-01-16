"""
Database Module - PostgreSQL/Neon DB Integration
Handles user authentication and long-term memory storage
"""

import os
import hashlib
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import uuid

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor, Json
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
    print("⚠️ psycopg2 not installed. Run: pip install psycopg2-binary")


class DatabaseManager:
    """
    PostgreSQL Database Manager for user authentication and long-term memory.
    Supports Neon DB and other PostgreSQL providers.
    """
    
    def __init__(
        self,
        database_url: str = None,
        host: str = None,
        port: int = 5432,
        database: str = None,
        user: str = None,
        password: str = None,
        sslmode: str = "require"
    ):
        """
        Initialize database connection.
        
        Args:
            database_url: Full connection URL (preferred for Neon DB)
            host, port, database, user, password: Individual connection params
            sslmode: SSL mode for connection (default: require for Neon)
        """
        self.conn = None
        self.database_url = database_url or os.getenv("DATABASE_URL")
        
        if not POSTGRES_AVAILABLE:
            print("❌ PostgreSQL driver not available")
            return
        
        # Connect using URL or individual params
        try:
            if self.database_url:
                # If URL already contains connection params, use it directly
                # This handles Neon DB URLs with sslmode and channel_binding
                self.conn = psycopg2.connect(self.database_url)
            else:
                self.conn = psycopg2.connect(
                    host=host or os.getenv("POSTGRES_HOST"),
                    port=port or int(os.getenv("POSTGRES_PORT", 5432)),
                    database=database or os.getenv("POSTGRES_DB"),
                    user=user or os.getenv("POSTGRES_USER"),
                    password=password or os.getenv("POSTGRES_PASSWORD"),
                    sslmode=sslmode
                )
            
            self.conn.autocommit = True
            print("✅ Connected to PostgreSQL database")
            
            # Initialize tables
            self._create_tables()
            
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            self.conn = None
    
    def _create_tables(self):
        """Create necessary tables if they don't exist"""
        if not self.conn:
            return
        
        with self.conn.cursor() as cur:
            # Users table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    user_id VARCHAR(50) UNIQUE NOT NULL,
                    username VARCHAR(100) UNIQUE NOT NULL,
                    password_hash VARCHAR(256) NOT NULL,
                    name VARCHAR(200) NOT NULL,
                    email VARCHAR(200),
                    role VARCHAR(100) DEFAULT 'User',
                    allowed_domains TEXT[] DEFAULT ARRAY['IT Service Desk'],
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP,
                    is_active BOOLEAN DEFAULT TRUE
                )
            """)
            
            # Long-term memory table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS long_term_memory (
                    id SERIAL PRIMARY KEY,
                    memory_id VARCHAR(64) UNIQUE NOT NULL,
                    user_id VARCHAR(50) NOT NULL,
                    domain VARCHAR(100),
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    embedding FLOAT8[],
                    importance_score FLOAT DEFAULT 0.5,
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
                )
            """)
            
            # Create index for faster retrieval
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_memory_user_id 
                ON long_term_memory(user_id)
            """)
            
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_memory_domain 
                ON long_term_memory(domain)
            """)
            
            # Sessions table for tracking
            cur.execute("""
                CREATE TABLE IF NOT EXISTS user_sessions (
                    id SERIAL PRIMARY KEY,
                    session_id VARCHAR(100) NOT NULL,
                    user_id VARCHAR(50) NOT NULL,
                    domain VARCHAR(100),
                    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    message_count INTEGER DEFAULT 0,
                    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
                )
            """)
            
            # Chat history table for persistence
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chat_history (
                    id SERIAL PRIMARY KEY,
                    user_id VARCHAR(50) NOT NULL,
                    domain VARCHAR(100) NOT NULL,
                    session_name VARCHAR(200) NOT NULL,
                    messages JSONB NOT NULL DEFAULT '[]',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
                    UNIQUE(user_id, domain, session_name)
                )
            """)
            
            # Index for faster chat retrieval
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_chat_user_domain 
                ON chat_history(user_id, domain)
            """)
            
            print("   ✓ Database tables initialized")
    
    def is_connected(self) -> bool:
        """Check if database is connected"""
        return self.conn is not None
    
    # ==================== USER AUTHENTICATION ====================
    
    def _hash_password(self, password: str) -> str:
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def create_user(
        self,
        username: str,
        password: str,
        name: str,
        email: str = None,
        role: str = "User",
        allowed_domains: List[str] = None
    ) -> Dict[str, Any]:
        """
        Create a new user account.
        
        Returns:
            Dict with success status and user info or error message
        """
        if not self.conn:
            return {"success": False, "error": "Database not connected"}
        
        try:
            user_id = f"EMP_{str(uuid.uuid4())[:8].upper()}"
            password_hash = self._hash_password(password)
            domains = allowed_domains or ["IT Service Desk"]
            
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    INSERT INTO users (user_id, username, password_hash, name, email, role, allowed_domains)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING user_id, username, name, email, role, allowed_domains, created_at
                """, (user_id, username, password_hash, name, email, role, domains))
                
                user = dict(cur.fetchone())
                user['id'] = user['user_id']
                
                return {
                    "success": True,
                    "user": user,
                    "message": f"User {username} created successfully"
                }
                
        except psycopg2.errors.UniqueViolation:
            return {"success": False, "error": "Username already exists"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def authenticate_user(self, username: str, password: str) -> Dict[str, Any]:
        """
        Authenticate user credentials.
        
        Returns:
            Dict with success status and user info or error message
        """
        if not self.conn:
            return {"success": False, "error": "Database not connected"}
        
        try:
            password_hash = self._hash_password(password)
            
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT user_id, username, name, email, role, allowed_domains, is_active
                    FROM users
                    WHERE username = %s AND password_hash = %s
                """, (username, password_hash))
                
                result = cur.fetchone()
                
                if result:
                    if not result['is_active']:
                        return {"success": False, "error": "Account is deactivated"}
                    
                    # Update last login
                    cur.execute("""
                        UPDATE users SET last_login = CURRENT_TIMESTAMP
                        WHERE username = %s
                    """, (username,))
                    
                    user = dict(result)
                    user['id'] = user['user_id']
                    
                    return {
                        "success": True,
                        "user": user
                    }
                else:
                    return {"success": False, "error": "Invalid username or password"}
                    
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_user_by_id(self, user_id: str) -> Optional[Dict]:
        """Get user by user_id"""
        if not self.conn:
            return None
        
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT user_id, username, name, email, role, allowed_domains
                    FROM users WHERE user_id = %s
                """, (user_id,))
                
                result = cur.fetchone()
                if result:
                    user = dict(result)
                    user['id'] = user['user_id']
                    return user
                return None
        except:
            return None
    
    def update_user_domains(self, user_id: str, domains: List[str]) -> bool:
        """Update user's allowed domains"""
        if not self.conn:
            return False
        
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    UPDATE users SET allowed_domains = %s
                    WHERE user_id = %s
                """, (domains, user_id))
                return True
        except:
            return False
    
    # ==================== LONG-TERM MEMORY ====================
    
    def store_memory(
        self,
        user_id: str,
        question: str,
        answer: str,
        domain: str = "general",
        embedding: List[float] = None,
        importance_score: float = 0.5,
        metadata: Dict = None
    ) -> Optional[str]:
        """
        Store a conversation to long-term memory.
        
        Returns:
            Memory ID if successful, None otherwise
        """
        if not self.conn:
            return None
        
        try:
            memory_id = hashlib.md5(
                f"{user_id}|{question}|{datetime.now().isoformat()}".encode()
            ).hexdigest()
            
            with self.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO long_term_memory 
                    (memory_id, user_id, domain, question, answer, embedding, importance_score, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (memory_id) DO UPDATE SET
                        answer = EXCLUDED.answer,
                        importance_score = EXCLUDED.importance_score,
                        metadata = EXCLUDED.metadata
                    RETURNING memory_id
                """, (
                    memory_id, user_id, domain, question, answer,
                    embedding, importance_score, Json(metadata or {})
                ))
                
                return memory_id
        except Exception as e:
            print(f"Error storing memory: {e}")
            return None
    
    def retrieve_memories(
        self,
        user_id: str,
        domain: str = None,
        limit: int = 10,
        min_importance: float = 0.0
    ) -> List[Dict]:
        """
        Retrieve memories for a user.
        
        Args:
            user_id: User identifier
            domain: Optional domain filter
            limit: Maximum number of memories to return
            min_importance: Minimum importance score filter
        """
        if not self.conn:
            return []
        
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                if domain:
                    cur.execute("""
                        SELECT memory_id, question, answer, domain, importance_score, created_at, metadata
                        FROM long_term_memory
                        WHERE user_id = %s AND domain = %s AND importance_score >= %s
                        ORDER BY created_at DESC
                        LIMIT %s
                    """, (user_id, domain, min_importance, limit))
                else:
                    cur.execute("""
                        SELECT memory_id, question, answer, domain, importance_score, created_at, metadata
                        FROM long_term_memory
                        WHERE user_id = %s AND importance_score >= %s
                        ORDER BY created_at DESC
                        LIMIT %s
                    """, (user_id, min_importance, limit))
                
                return [dict(row) for row in cur.fetchall()]
        except Exception as e:
            print(f"Error retrieving memories: {e}")
            return []
    
    def search_memories_by_text(
        self,
        user_id: str,
        search_text: str,
        domain: str = None,
        limit: int = 5
    ) -> List[Dict]:
        """
        Search memories by text content (simple text matching).
        For semantic search, use the embedding-based search in memory_manager.
        """
        if not self.conn:
            return []
        
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                search_pattern = f"%{search_text}%"
                
                if domain:
                    cur.execute("""
                        SELECT memory_id, question, answer, domain, importance_score, created_at
                        FROM long_term_memory
                        WHERE user_id = %s AND domain = %s
                        AND (question ILIKE %s OR answer ILIKE %s)
                        ORDER BY importance_score DESC, created_at DESC
                        LIMIT %s
                    """, (user_id, domain, search_pattern, search_pattern, limit))
                else:
                    cur.execute("""
                        SELECT memory_id, question, answer, domain, importance_score, created_at
                        FROM long_term_memory
                        WHERE user_id = %s
                        AND (question ILIKE %s OR answer ILIKE %s)
                        ORDER BY importance_score DESC, created_at DESC
                        LIMIT %s
                    """, (user_id, search_pattern, search_pattern, limit))
                
                return [dict(row) for row in cur.fetchall()]
        except Exception as e:
            print(f"Error searching memories: {e}")
            return []
    
    def delete_memory(self, memory_id: str, user_id: str) -> bool:
        """Delete a specific memory"""
        if not self.conn:
            return False
        
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    DELETE FROM long_term_memory
                    WHERE memory_id = %s AND user_id = %s
                """, (memory_id, user_id))
                return True
        except:
            return False
    
    def clear_user_memories(self, user_id: str, domain: str = None) -> bool:
        """Clear all memories for a user (optionally filtered by domain)"""
        if not self.conn:
            return False
        
        try:
            with self.conn.cursor() as cur:
                if domain:
                    cur.execute("""
                        DELETE FROM long_term_memory
                        WHERE user_id = %s AND domain = %s
                    """, (user_id, domain))
                else:
                    cur.execute("""
                        DELETE FROM long_term_memory
                        WHERE user_id = %s
                    """, (user_id,))
                return True
        except:
            return False
    
    def get_memory_stats(self, user_id: str) -> Dict[str, Any]:
        """Get memory statistics for a user"""
        if not self.conn:
            return {"total": 0, "by_domain": {}}
        
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Total count
                cur.execute("""
                    SELECT COUNT(*) as total FROM long_term_memory
                    WHERE user_id = %s
                """, (user_id,))
                total = cur.fetchone()['total']
                
                # Count by domain
                cur.execute("""
                    SELECT domain, COUNT(*) as count
                    FROM long_term_memory
                    WHERE user_id = %s
                    GROUP BY domain
                """, (user_id,))
                
                by_domain = {row['domain']: row['count'] for row in cur.fetchall()}
                
                return {
                    "total": total,
                    "by_domain": by_domain
                }
        except:
            return {"total": 0, "by_domain": {}}
    
    # ==================== SESSION TRACKING ====================
    
    def create_session(self, session_id: str, user_id: str, domain: str) -> bool:
        """Create or update a session record"""
        if not self.conn:
            return False
        
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO user_sessions (session_id, user_id, domain)
                    VALUES (%s, %s, %s)
                    ON CONFLICT DO NOTHING
                """, (session_id, user_id, domain))
                return True
        except:
            return False
    
    def update_session_activity(self, session_id: str, user_id: str) -> bool:
        """Update session last activity and increment message count"""
        if not self.conn:
            return False
        
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    UPDATE user_sessions
                    SET last_activity = CURRENT_TIMESTAMP, message_count = message_count + 1
                    WHERE session_id = %s AND user_id = %s
                """, (session_id, user_id))
                return True
        except:
            return False
    
    # ==================== CHAT HISTORY PERSISTENCE ====================
    
    def save_chat(self, user_id: str, domain: str, session_name: str, messages: List[Dict]) -> bool:
        """
        Save or update chat history for a user.
        
        Args:
            user_id: User identifier
            domain: Chat domain (IT Service Desk, Developer Support, HR Operations)
            session_name: Name of the chat session
            messages: List of message dicts with 'role' and 'content'
            
        Returns:
            True if successful
        """
        if not self.conn:
            return False
        
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO chat_history (user_id, domain, session_name, messages, updated_at)
                    VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)
                    ON CONFLICT (user_id, domain, session_name)
                    DO UPDATE SET messages = %s, updated_at = CURRENT_TIMESTAMP
                """, (user_id, domain, session_name, Json(messages), Json(messages)))
                return True
        except Exception as e:
            print(f"Error saving chat: {e}")
            return False
    
    def load_user_chats(self, user_id: str) -> Dict[str, Dict[str, List[Dict]]]:
        """
        Load all chat history for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dict structured as {domain: {session_name: [messages]}}
        """
        if not self.conn:
            return {}
        
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT domain, session_name, messages
                    FROM chat_history
                    WHERE user_id = %s
                    ORDER BY updated_at DESC
                """, (user_id,))
                
                rows = cur.fetchall()
                
                # Structure: {domain: {session_name: messages}}
                chats = {}
                for row in rows:
                    domain = row['domain']
                    session_name = row['session_name']
                    messages = row['messages'] if row['messages'] else []
                    
                    if domain not in chats:
                        chats[domain] = {}
                    chats[domain][session_name] = messages
                
                return chats
        except Exception as e:
            print(f"Error loading chats: {e}")
            return {}
    
    def delete_chat(self, user_id: str, domain: str, session_name: str) -> bool:
        """Delete a specific chat session"""
        if not self.conn:
            return False
        
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    DELETE FROM chat_history
                    WHERE user_id = %s AND domain = %s AND session_name = %s
                """, (user_id, domain, session_name))
                return True
        except Exception as e:
            print(f"Error deleting chat: {e}")
            return False
    
    def rename_chat(self, user_id: str, domain: str, old_name: str, new_name: str) -> bool:
        """Rename a chat session"""
        if not self.conn:
            return False
        
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    UPDATE chat_history
                    SET session_name = %s, updated_at = CURRENT_TIMESTAMP
                    WHERE user_id = %s AND domain = %s AND session_name = %s
                """, (new_name, user_id, domain, old_name))
                return True
        except Exception as e:
            print(f"Error renaming chat: {e}")
            return False
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            print("Database connection closed")


# Singleton instance
_db_instance = None

def get_database(database_url: str = None) -> DatabaseManager:
    """Get or create DatabaseManager singleton"""
    global _db_instance
    if _db_instance is None:
        _db_instance = DatabaseManager(database_url=database_url)
    return _db_instance
