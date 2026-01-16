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

st.set_page_config(
    page_title="HCLTech Enterprise Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

def process_query(query: str, domain: str, user_id: str, session_id: str, agent, tools, mode: str = "chat"):
    """
    Process user query through the agent pipeline.
    
    Args:
        mode: "chat" for PDF Q&A with RAG, "action" for tool JSON output
    """
    result = {
        "answer": "",
        "tool_calls": [],
        "reasoning_steps": [],
        "documents": [],
        "is_grounded": False,
        "tool_result": {},
        "action_json": None  # For action mode
    }
    
    if mode == "action":
        # ACTION MODE: Detect action and return JSON (no execution)
        result["reasoning_steps"].append("🎯 Action Mode: Detecting action from command...")
        
        if agent:
            try:
                # Use the agent's tool detection chain
                raw_res = agent.tool_chain.invoke({
                    "domain": domain,
                    "question": query,
                    "context": f"User ID: {user_id}, Domain: {domain}"
                })
                
                result["reasoning_steps"].append(f"   Raw LLM response: {raw_res[:100]}...")
                
                # Parse the JSON response
                import ast
                text = raw_res.strip().replace("```json", "").replace("```", "").strip()
                
                try:
                    tool_info = json.loads(text)
                except json.JSONDecodeError:
                    try:
                        tool_info = ast.literal_eval(text)
                    except:
                        # Try to extract JSON from the response
                        import re
                        json_match = re.search(r'\{[^{}]*\}', text)
                        if json_match:
                            tool_info = json.loads(json_match.group())
                        else:
                            tool_info = {"tool": "unknown", "parameters": {"raw_query": query}}
                
                # Build the action JSON response
                action = tool_info.get("tool", "unknown")
                params = tool_info.get("parameters", {})
                
                action_json = {
                    "action": action,
                    "parameters": params,
                    "metadata": {
                        "requester_id": user_id,
                        "domain": domain,
                        "timestamp": datetime.now().isoformat(),
                        "status": "pending"
                    }
                }
                
                # Add API endpoint based on action type
                if action in ["create_ticket", "password_reset", "software_request", "troubleshoot", "system_status"]:
                    action_json["api"] = {
                        "service": "IT Service Desk",
                        "endpoint": f"/api/v1/it/{action}",
                        "method": "POST"
                    }
                elif action in ["schedule_meeting", "leave_application", "policy_query", "benefits_info", "payroll_query", "employee_lookup"]:
                    action_json["api"] = {
                        "service": "HR Operations",
                        "endpoint": f"/api/v1/hr/{action}",
                        "method": "POST"
                    }
                elif action in ["code_review", "api_docs", "deploy_request"]:
                    action_json["api"] = {
                        "service": "Developer Support",
                        "endpoint": f"/api/v1/dev/{action}",
                        "method": "POST"
                    }
                else:
                    action_json["api"] = {
                        "service": "General",
                        "endpoint": f"/api/v1/actions/{action}",
                        "method": "POST"
                    }
                
                result["action_json"] = action_json
                result["tool_calls"] = [tool_info]
                
                # Format answer as clean JSON display
                result["answer"] = f"```json\n{json.dumps(action_json, indent=2)}\n```"
                result["reasoning_steps"].append(f"   ✅ Action detected: {action}")
                    
            except Exception as e:
                import traceback
                error_detail = traceback.format_exc()
                result["reasoning_steps"].append(f"   ❌ Error: {str(e)}")
                result["reasoning_steps"].append(f"   Details: {error_detail[-200:]}")
                
                # Return error as JSON
                error_json = {
                    "error": str(e),
                    "query": query,
                    "status": "failed"
                }
                result["action_json"] = error_json
                result["answer"] = f"```json\n{json.dumps(error_json, indent=2)}\n```"
        else:
            error_json = {"error": "Agent not initialized", "status": "failed"}
            result["action_json"] = error_json
            result["answer"] = f"```json\n{json.dumps(error_json, indent=2)}\n```"
        
        return result
    
    # CHAT MODE: Full RAG pipeline with memory (no tool execution)
    if agent:
        try:
            result["reasoning_steps"].append(f"📤 Chat Mode: Querying knowledge base...")
            agent_result = agent.invoke(
                question=query,
                domain=domain,
                user_id=user_id,
                session_id=session_id
            )
            
            if agent_result:
                result.update(agent_result)
                result["reasoning_steps"].append(f"✅ Retrieved {len(agent_result.get('documents', []))} documents")
            else:
                result["answer"] = "I couldn't find relevant information. Please try rephrasing your question."
                result["reasoning_steps"].append("⚠️ No results from agent")
            
        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            result["answer"] = f"Error processing query: {str(e)}"
            result["reasoning_steps"].append(f"❌ Error: {str(e)}")
            print(f"Process query error: {error_detail}")
    else:
        result["answer"] = f"Agent not initialized. Please check your GROQ_API_KEY configuration.\n\nYour query: {query}"
        result["reasoning_steps"] = ["⚠️ Agent not available"]
    
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
                        
                        # Load previous chats from database
                        if db and db.is_connected():
                            saved_chats = db.load_user_chats(user['id'])
                            if saved_chats:
                                max_chat_num = 0
                                for domain, sessions in saved_chats.items():
                                    if domain in st.session_state.domain_chats:
                                        st.session_state.domain_chats[domain].update(sessions)
                                    else:
                                        st.session_state.domain_chats[domain] = sessions
                                    
                                    # Track highest chat number for counter
                                    for session_name in sessions.keys():
                                        if session_name.startswith("Chat "):
                                            try:
                                                num = int(session_name.split(" ")[1])
                                                max_chat_num = max(max_chat_num, num)
                                            except:
                                                pass
                                
                                # Set counter to continue from highest
                                st.session_state.global_session_counter = max_chat_num
                        
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
            
            # Debug search feature
            with st.form("debug_search_form"):
                debug_query = st.text_input("Test search query", value="help")
                if st.form_submit_button("🔍 Debug Search"):
                    debug_result = engine.debug_search(debug_query)
                    st.json(debug_result)
        
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
        
        # Save empty chat to database immediately
        if db and db.is_connected():
            db.save_chat(
                user_id=user['id'],
                domain=st.session_state.current_domain,
                session_name=new_session_name,
                messages=[]
            )
        
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
                        
                        # Update in database
                        if db and db.is_connected():
                            db.rename_chat(
                                user_id=user['id'],
                                domain=st.session_state.current_domain,
                                old_name=selected_session,
                                new_name=new_name
                            )
                        
                        st.rerun()
            
            st.markdown("---")
            if st.button("🗑️ Delete Chat", type="secondary"):
                del st.session_state.domain_chats[st.session_state.current_domain][selected_session]
                
                # Delete from database
                if db and db.is_connected():
                    db.delete_chat(
                        user_id=user['id'],
                        domain=st.session_state.current_domain,
                        session_name=selected_session
                    )
                
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
    # Mode selector at the top
    mode_col1, mode_col2 = st.columns([3, 1])
    with mode_col1:
        st.subheader(f"{domain_icons.get(st.session_state.current_domain, '🤖')} HCLTech Enterprise Assistant")
    with mode_col2:
        chat_mode = st.toggle("🎯 Action Mode", value=False, help="Toggle for Action commands (returns JSON)")
    
    # Show current mode indicator
    if chat_mode:
        st.info("🎯 **Action Mode**: Give commands like 'Schedule a meeting with HR' - I'll show the API JSON")
    else:
        st.success("📄 **Chat Mode**: Ask questions about the HCLTech Annual Report - I'll answer from the document")
    
    if st.session_state.active_session_id and \
       st.session_state.active_session_id in st.session_state.domain_chats[st.session_state.current_domain]:
        
        active_history = st.session_state.domain_chats[st.session_state.current_domain][st.session_state.active_session_id]
        
        # Chat Container
        chat_container = st.container(height=450, border=True)
        with chat_container:
            if not active_history:
                st.caption(f"🚀 Started new conversation in **{st.session_state.current_domain}**")
                
                # Mode-specific welcome messages
                if chat_mode:
                    st.markdown("""
                    **Action Mode Examples:**
                    - "Schedule a meeting with HR for tomorrow"
                    - "Create a support ticket for my laptop not starting"
                    - "Reset my password"
                    - "Request installation of Docker"
                    - "Apply for leave from Jan 20 to Jan 25"
                    """)
                else:
                    st.markdown("""
                    **Chat with PDF Examples:**
                    - "What are the key risks mentioned on page 45?"
                    - "Summarize the financial highlights"
                    - "What is mentioned about AI initiatives?"
                    - "Tell me about page 10"
                    - "What are the company's ESG goals?"
                    """)
            
            for message in active_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
        
        # Input Area
        input_placeholder = "Give a command..." if chat_mode else "Ask about the HCLTech Annual Report..."
        prompt = st.chat_input(input_placeholder)
        
        if prompt:
            # Add user message
            st.session_state.domain_chats[st.session_state.current_domain][st.session_state.active_session_id].append(
                {"role": "user", "content": prompt}
            )
            
            # Process query with mode
            with st.spinner("🤔 Processing your request..."):
                response = process_query(
                    query=prompt,
                    domain=st.session_state.current_domain,
                    user_id=user['id'],
                    session_id=st.session_state.active_session_id,
                    agent=agent,
                    tools=tools,
                    mode="action" if chat_mode else "chat"
                )
            
            # Store response for inspector
            st.session_state.last_response = response
            
            # Add assistant response
            st.session_state.domain_chats[st.session_state.current_domain][st.session_state.active_session_id].append(
                {"role": "assistant", "content": response["answer"]}
            )
            
            # Save chat to database for persistence
            if db and db.is_connected():
                db.save_chat(
                    user_id=user['id'],
                    domain=st.session_state.current_domain,
                    session_name=st.session_state.active_session_id,
                    messages=st.session_state.domain_chats[st.session_state.current_domain][st.session_state.active_session_id]
                )
            
            st.rerun()
    else:
        st.info(f"👈 Click '+ New Chat' to start a conversation.")

# --- RIGHT PANEL: INSPECTOR ---
with col_inspector:
    st.subheader("🔍 Inspector Panel")
    
    if st.session_state.active_session_id:
        tab1, tab2, tab3 = st.tabs(["📋 Action JSON", "📄 Source Documents", "🧠 Reasoning"])
        
        with tab1:
            st.caption("**Tool/Action Detection**")
            
            if st.session_state.last_response and st.session_state.last_response.get("action_json"):
                # Action Mode - show the full action JSON
                st.success("✅ Action Detected!")
                action_json = st.session_state.last_response["action_json"]
                st.json(action_json)
                
                # Copy button
                st.code(json.dumps(action_json, indent=2), language="json")
                
            elif st.session_state.last_response and st.session_state.last_response.get("tool_calls"):
                # Show detected tool calls
                for tc in st.session_state.last_response["tool_calls"]:
                    if tc.get("tool") and tc["tool"] != "none":
                        st.info(f"🎯 Detected: **{tc['tool']}**")
                        st.json({
                            "action": tc["tool"],
                            "parameters": tc.get("parameters", {}),
                            "service": f"{st.session_state.current_domain} API",
                            "status": "detected"
                        })
                    else:
                        st.caption("No action detected for this query")
            else:
                # Default placeholder
                st.caption("No action detected. Try Action Mode for commands like:")
                st.markdown("""
                - "Schedule a meeting with HR"
                - "Create a ticket for laptop issue"
                - "Reset my password"
                """)
        
        with tab2:
            st.caption("**Retrieved Context from PDF**")
            
            if st.session_state.last_response and st.session_state.last_response.get("documents"):
                documents = st.session_state.last_response["documents"]
                
                if isinstance(documents, list) and len(documents) > 0:
                    # Summary stats
                    st.metric("Documents Retrieved", len(documents))
                    
                    for i, doc_info in enumerate(documents[:5], 1):
                        if isinstance(doc_info, dict):
                            content = doc_info.get("content", str(doc_info))
                            source = doc_info.get("source", "HCLTech Report")
                            doc_type = doc_info.get("type", "text")
                            metadata = doc_info.get("metadata", {})
                            page = metadata.get("page", "N/A")
                        else:
                            content = str(doc_info)
                            source = "HCLTech Report"
                            page = "N/A"
                            metadata = {}
                        
                        # Create expandable section for each doc
                        with st.expander(f"📄 Source {i} - Page {page}", expanded=(i == 1)):
                            if metadata:
                                col_a, col_b = st.columns(2)
                                with col_a:
                                    st.caption(f"**Page:** {page}")
                                with col_b:
                                    st.caption(f"**Type:** {metadata.get('type', 'text')}")
                            
                            # Show content preview
                            if len(content) > 500:
                                st.text_area(
                                    "Content",
                                    content[:500] + "\n...[truncated]",
                                    height=120,
                                    disabled=True,
                                    key=f"doc_{i}"
                                )
                            else:
                                st.text_area("Content", content, height=100, disabled=True, key=f"doc_{i}")
                    
                    # Grounding indicator
                    st.divider()
                    is_grounded = st.session_state.last_response.get("is_grounded", False)
                    if is_grounded:
                        st.success("✅ Answer grounded in source documents")
                    else:
                        st.warning("⚠️ Answer may include general knowledge")
                else:
                    st.info("No documents retrieved for this query.")
            else:
                st.info("Ask a question about the PDF to see retrieved context.")
                st.caption("**Example queries:**")
                st.markdown("""
                - "What are the key risks on page 45?"
                - "Summarize the financial highlights"
                - "What is HCL's AI strategy?"
                """)
        
        with tab3:
            st.caption("**Agent Reasoning Steps**")
            
            if st.session_state.last_response and st.session_state.last_response.get("reasoning_steps"):
                with st.container(height=350):
                    for step in st.session_state.last_response["reasoning_steps"]:
                        if "✅" in step or "✓" in step:
                            st.success(step)
                        elif "❌" in step or "⚠️" in step:
                            st.warning(step)
                        else:
                            st.text(step)
            else:
                st.info("Reasoning steps will appear here after processing a query.")
                st.caption("The agent shows:")
                st.markdown("""
                - 🧠 Memory retrieval
                - 🔍 Document search
                - 📝 Relevance grading
                - 💡 Answer generation
                - 🛡️ Grounding verification
                """)
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
