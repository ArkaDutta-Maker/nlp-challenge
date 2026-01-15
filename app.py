"""
ByteMe Enterprise Assistant - Streamlit Application
Agentic RAG with Long-term (PostgreSQL/ChromaDB) and Short-term (Redis) Memory
User Authentication via PostgreSQL (Neon DB)
"""

import streamlit as st
import json
import time
import os
from datetime import datetime

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------
# 1. PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="HCLTech Enterprise Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# 2. LAZY LOADING & INITIALIZATION
# ---------------------------------------------------------

@st.cache_resource
def initialize_database():
    """Initialize PostgreSQL Database (cached)"""
    try:
        from database import get_database
        database_url = os.getenv("DATABASE_URL")
        db = get_database(database_url=database_url)
        return db
    except Exception as e:
        st.warning(f"PostgreSQL not available: {e}")
        return None

@st.cache_resource
def initialize_engine():
    """Initialize the ByteMe Engine (cached)"""
    try:
        from engine import get_engine
        engine = get_engine(db_path="./chroma_db", use_vision=False)
        return engine
    except Exception as e:
        st.error(f"Failed to initialize engine: {e}")
        return None

@st.cache_resource
def initialize_memory_manager(_engine, _db):
    """Initialize the Memory Manager (cached)"""
    if _engine is None:
        return None
    try:
        from memory_manager import get_memory_manager
        
        # Get Redis config from environment or use defaults
        redis_host = os.getenv("REDIS_HOST", "localhost")
        redis_port = int(os.getenv("REDIS_PORT", 6379))
        redis_password = os.getenv("REDIS_PASSWORD", None)
        
        # Use PostgreSQL for long-term memory if available
        memory_manager = get_memory_manager(
            chromadb_client=_engine.client,
            embedder=_engine.dense_embedder,
            db_manager=_db,
            redis_host=redis_host,
            redis_port=redis_port,
            redis_password=redis_password,
            use_postgres_memory=(_db is not None and _db.is_connected())
        )
        return memory_manager
    except Exception as e:
        st.error(f"Failed to initialize memory manager: {e}")
        return None

@st.cache_resource
def initialize_agent(_engine, _memory_manager):
    """Initialize the ByteMe Agent (cached)"""
    if _engine is None or _memory_manager is None:
        return None
    try:
        from agent import get_agent
        
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            st.warning("⚠️ GROQ_API_KEY not set. Agent functionality will be limited.")
            return None
        
        agent = get_agent(
            engine=_engine,
            memory_manager=_memory_manager,
            groq_api_key=groq_api_key
        )
        return agent
    except Exception as e:
        st.error(f"Failed to initialize agent: {e}")
        return None

@st.cache_resource
def initialize_tools():
    """Initialize domain-specific tools (cached)"""
    try:
        from tools import ITServiceDeskTool, DeveloperSupportTool, HROperationsTool
        return {
            "IT Service Desk": ITServiceDeskTool(),
            "Developer Support": DeveloperSupportTool(),
            "HR Operations": HROperationsTool()
        }
    except Exception as e:
        st.warning(f"Failed to initialize tools: {e}")
        return {}

# ---------------------------------------------------------
# 3. HELPER FUNCTIONS
# ---------------------------------------------------------

def load_creds():
    """Loads user credentials from cred.json (fallback)"""
    try:
        if not os.path.exists("cred.json"):
            return {}
        with open("cred.json", "r") as f:
            return json.load(f)
    except Exception as e:
        return {}

def authenticate_json(username, password, data):
    """Verifies username and password against JSON data (fallback)"""
    if "users" in data and username in data["users"]:
        if data["users"][username]["password"] == password:
            return data["users"][username]
    return None

def authenticate_user(username: str, password: str, db):
    """
    Authenticate user - PostgreSQL first, then JSON fallback.
    """
    # Try PostgreSQL authentication first
    if db and db.is_connected():
        result = db.authenticate_user(username, password)
        if result["success"]:
            return result["user"]
    
    # Fallback to JSON file
    creds = load_creds()
    return authenticate_json(username, password, creds)

def register_user(username: str, password: str, name: str, email: str, role: str, domains: list, db):
    """
    Register a new user in PostgreSQL.
    """
    if not db or not db.is_connected():
        return {"success": False, "error": "Database not connected"}
    
    return db.create_user(
        username=username,
        password=password,
        name=name,
        email=email,
        role=role,
        allowed_domains=domains
    )

def process_query(query: str, domain: str, user_id: str, session_id: str, agent, tools):
    """Process user query through the agent pipeline"""
    result = {
        "answer": "",
        "tool_calls": [],
        "reasoning_steps": [],
        "documents": [],
        "is_grounded": False
    }
    
    if agent:
        try:
            # Invoke the LangGraph agent
            agent_result = agent.invoke(
                question=query,
                domain=domain,
                user_id=user_id,
                session_id=session_id
            )
            result.update(agent_result)
            
            # Execute any detected tool calls
            if agent_result.get("tool_calls") and domain in tools:
                tool = tools[domain]
                for tc in agent_result["tool_calls"]:
                    if tc.get("tool") and tc["tool"] != "none":
                        tool_result = tool.execute_action(
                            action=tc["tool"],
                            parameters=tc.get("parameters", {})
                        )
                        tc["result"] = tool_result
                        
                        # Append tool result to answer if successful
                        if tool_result.get("success"):
                            result["answer"] += f"\n\n**Tool Action: {tc['tool']}**\n"
                            result["answer"] += f"```json\n{json.dumps(tool_result, indent=2, default=str)}\n```"
            
        except Exception as e:
            result["answer"] = f"Error processing query: {str(e)}"
            result["reasoning_steps"].append(f"❌ Error: {str(e)}")
    else:
        # Fallback mode without agent
        result["answer"] = f"Agent not initialized. Please check your GROQ_API_KEY configuration.\n\nYour query: {query}"
        result["reasoning_steps"] = ["⚠️ Agent not available - fallback mode"]
    
    return result

# ---------------------------------------------------------
# 4. SESSION STATE INITIALIZATION
# ---------------------------------------------------------

# Authentication State
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "user_info" not in st.session_state:
    st.session_state.user_info = {}
if "current_domain" not in st.session_state:
    st.session_state.current_domain = None

# Chat Data State - NESTED STRUCTURE
if "domain_chats" not in st.session_state:
    st.session_state.domain_chats = {}

if "active_session_id" not in st.session_state:
    st.session_state.active_session_id = None

if "global_session_counter" not in st.session_state:
    st.session_state.global_session_counter = 0

# Inspector panel state
if "last_response" not in st.session_state:
    st.session_state.last_response = None

# ---------------------------------------------------------
# 5. LOGIN SCREEN
# ---------------------------------------------------------

# Initialize database early for auth
db = initialize_database()

if not st.session_state.authenticated:
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1.5, 1])
    
    with col2:
        if os.path.exists("nlpc.jpg"):
            st.image("nlpc.jpg", width=200)
        else:
            st.title("🤖 Byte Me")
        
        # Tab for Sign In / Sign Up
        auth_tab = st.tabs(["🔐 Sign In", "📝 Sign Up"])
        
        # SIGN IN TAB
        with auth_tab[0]:
            with st.form("login_form"):
                st.subheader("Sign In")
                username = st.text_input("Username", key="login_username")
                password = st.text_input("Password", type="password", key="login_password")
                
                submit = st.form_submit_button("Sign In", use_container_width=True)
                
                if submit:
                    user = authenticate_user(username, password, db)
                    
                    if user:
                        st.session_state.authenticated = True
                        st.session_state.user_info = user
                        
                        # Initialize domain chat storage
                        for domain in user["allowed_domains"]:
                            if domain not in st.session_state.domain_chats:
                                st.session_state.domain_chats[domain] = {}
                        
                        # Set default domain
                        if user["allowed_domains"]:
                            st.session_state.current_domain = user["allowed_domains"][0]
                        
                        st.success(f"Welcome back, {user['name']}!")
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        st.error("Invalid Username or Password")
            
            # Show database status
            if db and db.is_connected():
                st.caption("🟢 Connected to PostgreSQL")
            else:
                st.caption("🟡 Using local authentication")
        
        # SIGN UP TAB
        with auth_tab[1]:
            if db and db.is_connected():
                with st.form("signup_form"):
                    st.subheader("Create Account")
                    
                    new_username = st.text_input("Username", key="signup_username")
                    new_password = st.text_input("Password", type="password", key="signup_password")
                    confirm_password = st.text_input("Confirm Password", type="password", key="signup_confirm")
                    new_name = st.text_input("Full Name", key="signup_name")
                    new_email = st.text_input("Email", key="signup_email")
                    
                    new_role = st.selectbox(
                        "Role",
                        ["User", "IT-Admin", "Developer", "HR Manager", "Lead Developer"],
                        key="signup_role"
                    )
                    
                    available_domains = ["IT Service Desk", "Developer Support", "HR Operations"]
                    selected_domains = st.multiselect(
                        "Access Domains",
                        available_domains,
                        default=["IT Service Desk"],
                        key="signup_domains"
                    )
                    
                    signup_submit = st.form_submit_button("Create Account", use_container_width=True)
                    
                    if signup_submit:
                        # Validation
                        if not all([new_username, new_password, new_name]):
                            st.error("Please fill in all required fields")
                        elif new_password != confirm_password:
                            st.error("Passwords do not match")
                        elif len(new_password) < 6:
                            st.error("Password must be at least 6 characters")
                        elif not selected_domains:
                            st.error("Please select at least one domain")
                        else:
                            result = register_user(
                                username=new_username,
                                password=new_password,
                                name=new_name,
                                email=new_email,
                                role=new_role,
                                domains=selected_domains,
                                db=db
                            )
                            
                            if result["success"]:
                                st.success(f"Account created! User ID: {result['user']['user_id']}")
                                st.info("Please sign in with your new credentials.")
                            else:
                                st.error(f"Registration failed: {result.get('error', 'Unknown error')}")
            else:
                st.info("📝 Sign up requires PostgreSQL database connection.")
                st.caption("Configure DATABASE_URL in your environment to enable registration.")
                
                with st.expander("Demo Accounts"):
                    st.markdown("""
                    | Username | Password | Role |
                    |----------|----------|------|
                    | john | password123 | IT-Admin |
                    | sarah | hrpass | HR Manager |
                    | mike | devpass | Developer |
                    """)
    
    st.stop()

# ---------------------------------------------------------
# 6. INITIALIZE COMPONENTS (POST-LOGIN)
# ---------------------------------------------------------

# Initialize components (db already initialized above)
engine = initialize_engine()
memory_manager = initialize_memory_manager(engine, db)
agent = initialize_agent(engine, memory_manager)
tools = initialize_tools()

user = st.session_state.user_info

# ---------------------------------------------------------
# 7. SIDEBAR
# ---------------------------------------------------------

with st.sidebar:
    if os.path.exists("nlpc.jpg"):
        st.image("nlpc.jpg", use_container_width=True)
    else:
        st.markdown("## 🤖 Byte Me")
    
    st.title("Enterprise Assistant")
    
    # System Status
    with st.expander("🔧 System Status", expanded=False):
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Engine", "✅" if engine else "❌")
            st.metric("Memory", "✅" if memory_manager else "❌")
        with col_b:
            st.metric("Agent", "✅" if agent else "❌")
            st.metric("Database", "✅" if db and db.is_connected() else "❌")
        
        if engine:
            stats = engine.get_stats()
            st.caption(f"📄 Documents: {stats['text_documents']}")
        
        if memory_manager:
            mem_stats = memory_manager.get_stats()
            st.caption(f"🧠 Short-term: {mem_stats['short_term']['backend']}")
            if 'long_term' in mem_stats:
                lt_backend = mem_stats['long_term'].get('backend', 'chromadb')
                st.caption(f"📦 Long-term: {lt_backend}")
        
        # User memory stats (PostgreSQL only)
        if db and db.is_connected():
            user_mem_stats = db.get_memory_stats(user['id'])
            st.caption(f"💾 Your memories: {user_mem_stats['total']}")
    
    st.divider()
    
    # 1. DOMAIN SWITCHER
    st.subheader("🎯 Domain Switcher")
    
    domain_icons = {
        "IT Service Desk": "🛠️",
        "Developer Support": "💻",
        "HR Operations": "👥"
    }
    
    if user["allowed_domains"]:
        if st.session_state.current_domain not in user["allowed_domains"]:
            st.session_state.current_domain = user["allowed_domains"][0]
        
        # Format domain options with icons
        domain_options = [f"{domain_icons.get(d, '📌')} {d}" for d in user["allowed_domains"]]
        current_idx = user["allowed_domains"].index(st.session_state.current_domain)
        
        selected = st.radio(
            "Select Mode:",
            domain_options,
            index=current_idx,
            key="domain_radio"
        )
        
        # Extract domain name without icon
        st.session_state.current_domain = user["allowed_domains"][domain_options.index(selected)]
    else:
        st.error("No domains assigned.")
    
    st.divider()
    
    # 2. USER PROFILE
    st.subheader("👤 User Profile")
    with st.container(border=True):
        st.markdown(f"**{user['name']}**")
        st.caption(user['role'])
        st.caption(f"ID: {user['id']}")
        domain_icon = domain_icons.get(st.session_state.current_domain, "📌")
        st.info(f"{domain_icon} {st.session_state.current_domain}")
    
    st.divider()
    
    # 3. SESSION MANAGEMENT
    st.subheader("💬 Session History")
    
    if st.session_state.current_domain not in st.session_state.domain_chats:
        st.session_state.domain_chats[st.session_state.current_domain] = {}
    
    current_domain_sessions = st.session_state.domain_chats[st.session_state.current_domain]
    session_keys = list(current_domain_sessions.keys())
    
    # "New Chat" Button
    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        st.session_state.global_session_counter += 1
        new_session_name = f"Chat {st.session_state.global_session_counter}"
        
        st.session_state.domain_chats[st.session_state.current_domain][new_session_name] = []
        st.session_state.active_session_id = new_session_name
        st.session_state.last_response = None
        st.rerun()
    
    # Session Selector
    if session_keys:
        if st.session_state.active_session_id not in session_keys:
            st.session_state.active_session_id = session_keys[-1]
        
        selected_session = st.radio(
            "Active Chats",
            session_keys,
            index=session_keys.index(st.session_state.active_session_id),
            key="session_select_radio"
        )
        st.session_state.active_session_id = selected_session
        
        # Manage Session Controls
        with st.expander("⚙️ Manage Session"):
            new_name = st.text_input("Rename Chat", value=selected_session)
            if st.button("Update Name"):
                if new_name and new_name != selected_session:
                    if new_name in current_domain_sessions:
                        st.error("Name already exists!")
                    else:
                        data = st.session_state.domain_chats[st.session_state.current_domain].pop(selected_session)
                        st.session_state.domain_chats[st.session_state.current_domain][new_name] = data
                        st.session_state.active_session_id = new_name
                        st.rerun()
            
            st.markdown("---")
            if st.button("🗑️ Delete Chat", type="secondary"):
                del st.session_state.domain_chats[st.session_state.current_domain][selected_session]
                st.session_state.active_session_id = None
                st.session_state.last_response = None
                st.rerun()
            
            if st.button("🧹 Clear Memory"):
                if memory_manager:
                    memory_manager.clear_session(
                        session_id=st.session_state.active_session_id,
                        user_id=user['id']
                    )
                    st.success("Session memory cleared!")
    else:
        st.session_state.active_session_id = None
        st.info(f"No active chats in {st.session_state.current_domain}.")
    
    # Logout
    st.divider()
    if st.button("🚪 Logout"):
        st.session_state.clear()
        st.rerun()

# ---------------------------------------------------------
# 8. MAIN CONTENT AREA
# ---------------------------------------------------------

col_chat, col_inspector = st.columns([0.65, 0.35], gap="medium")

# --- CENTER PANEL: CHAT ---
with col_chat:
    st.subheader(f"{domain_icons.get(st.session_state.current_domain, '🤖')} HCL-Tech Enterprise Assistant")
    
    if st.session_state.active_session_id and \
       st.session_state.active_session_id in st.session_state.domain_chats[st.session_state.current_domain]:
        
        active_history = st.session_state.domain_chats[st.session_state.current_domain][st.session_state.active_session_id]
        
        # Chat Container
        chat_container = st.container(height=500, border=True)
        with chat_container:
            if not active_history:
                st.caption(f"🚀 Started new conversation in **{st.session_state.current_domain}**")
                
                # Domain-specific welcome messages
                welcome_messages = {
                    "IT Service Desk": "I can help you with troubleshooting, creating support tickets, software requests, and password resets.",
                    "Developer Support": "I can assist with legacy code documentation, suggest code fixes, and provide API documentation.",
                    "HR Operations": "I can answer policy questions, guide you through leave applications, and provide benefits information."
                }
                st.info(welcome_messages.get(st.session_state.current_domain, "How can I help you today?"))
            
            for message in active_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
        
        # Input Area
        prompt = st.chat_input(f"Ask about {st.session_state.current_domain}...")
        
        if prompt:
            # Add user message
            st.session_state.domain_chats[st.session_state.current_domain][st.session_state.active_session_id].append(
                {"role": "user", "content": prompt}
            )
            
            # Process query
            with st.spinner("🤔 Processing your request..."):
                response = process_query(
                    query=prompt,
                    domain=st.session_state.current_domain,
                    user_id=user['id'],
                    session_id=st.session_state.active_session_id,
                    agent=agent,
                    tools=tools
                )
            
            # Store response for inspector
            st.session_state.last_response = response
            
            # Add assistant response
            st.session_state.domain_chats[st.session_state.current_domain][st.session_state.active_session_id].append(
                {"role": "assistant", "content": response["answer"]}
            )
            
            st.rerun()
    else:
        st.info(f"👈 Click '+ New Chat' to start a {st.session_state.current_domain} session.")

# --- RIGHT PANEL: INSPECTOR ---
with col_inspector:
    st.subheader("🔍 Inspector")
    
    if st.session_state.active_session_id:
        tab1, tab2, tab3 = st.tabs(["📋 Action JSON", "📄 Source Preview", "🧠 Reasoning"])
        
        with tab1:
            st.caption("Generated Payload")
            
            if st.session_state.last_response and st.session_state.last_response.get("tool_calls"):
                for tc in st.session_state.last_response["tool_calls"]:
                    st.json(tc)
            else:
                mock_payload = {
                    "tool_call": "none",
                    "requester_id": user['id'],
                    "domain_context": st.session_state.current_domain,
                    "session_id": st.session_state.active_session_id,
                    "timestamp": datetime.now().isoformat()
                }
                st.json(mock_payload)
        
        with tab2:
            st.caption("Retrieved Context")
            
            if st.session_state.last_response and st.session_state.last_response.get("documents"):
                for i, doc in enumerate(st.session_state.last_response["documents"][:3], 1):
                    with st.expander(f"📄 Document {i}"):
                        st.text(doc[:500] + "..." if len(doc) > 500 else doc)
            else:
                st.info("📄 No documents retrieved yet.")
            
            # Grounding status
            if st.session_state.last_response:
                is_grounded = st.session_state.last_response.get("is_grounded", False)
                if is_grounded:
                    st.success("✅ Response is grounded in source documents")
                else:
                    st.warning("⚠️ Response may not be fully grounded")
        
        with tab3:
            st.caption("Agent Reasoning Steps")
            
            if st.session_state.last_response and st.session_state.last_response.get("reasoning_steps"):
                with st.container(height=350):
                    for step in st.session_state.last_response["reasoning_steps"]:
                        st.text(step)
            else:
                with st.status("Agent Status", expanded=True):
                    st.write(f"🔍 Context: {st.session_state.current_domain}")
                    st.write(f"📂 Session: {st.session_state.active_session_id}")
                    st.write(f"👤 User: {user['role']}")
                    st.write("⏳ Waiting for query...")
    else:
        st.caption("Start a chat to see inspection details.")

# ---------------------------------------------------------
# 9. FOOTER
# ---------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.caption("© 2026 Byte Me | Kshitij NLP Challenge")

# Memory backend info
short_term_backend = "Redis" if memory_manager and memory_manager.short_term.use_redis else "In-Memory"
long_term_backend = "PostgreSQL" if (db and db.is_connected()) else "ChromaDB"
st.sidebar.caption(f"Memory: {short_term_backend} | {long_term_backend}")
