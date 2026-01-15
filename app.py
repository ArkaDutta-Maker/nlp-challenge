import streamlit as st
import json
import time
import os

# ---------------------------------------------------------
# 1. PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="HCLTech Enterprise Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# 2. HELPER FUNCTIONS
# ---------------------------------------------------------

def load_creds():
    """Loads user credentials from cred.json"""
    try:
        if not os.path.exists("cred.json"):
            st.error("cred.json file not found! Please ensure it exists in the app directory.")
            return {}
        with open("cred.json", "r") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error reading credentials: {e}")
        return {}

def authenticate(username, password, data):
    """Verifies username and password against the loaded JSON data"""
    if "users" in data and username in data["users"]:
        if data["users"][username]["password"] == password:
            return data["users"][username]
    return None

# ---------------------------------------------------------
# 3. SESSION STATE INITIALIZATION
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

# ---------------------------------------------------------
# 4. LOGIN SCREEN
# ---------------------------------------------------------

if not st.session_state.authenticated:
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        if os.path.exists("nlpc.jpg"):
            st.image("nlpc.jpg", width=200)
        else:
            st.title("Byte Me Login")
        
        with st.form("login_form"):
            st.subheader("Sign In")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            
            submit = st.form_submit_button("Sign In")
            
            if submit:
                creds = load_creds()
                user = authenticate(username, password, creds)
                
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
                    
                    st.rerun()
                else:
                    st.error("Invalid Username or Password")
    
    st.stop()

# ---------------------------------------------------------
# 5. MAIN APPLICATION (POST-LOGIN)
# ---------------------------------------------------------

user = st.session_state.user_info

# --- SIDEBAR LOGIC ---
with st.sidebar:
    if os.path.exists("nlpc.jpg"):
        st.image("nlpc.jpg", use_container_width=True)
    else:
        st.header("Byte Me")
    
    st.title("Byte Me")
    
    # 1. DOMAIN SWITCHER
    st.subheader("Domain Switcher")
    
    def on_domain_change():
        pass

    if user["allowed_domains"]:
        if st.session_state.current_domain not in user["allowed_domains"]:
             st.session_state.current_domain = user["allowed_domains"][0]

        selected_domain = st.radio(
            "Select Mode:",
            user["allowed_domains"],
            index=user["allowed_domains"].index(st.session_state.current_domain),
            key="domain_radio",
            on_change=on_domain_change
        )
        st.session_state.current_domain = selected_domain
    else:
        st.error("No domains assigned.")
    
    st.divider()

    # 2. USER PROFILE
    st.subheader("User Profile")
    with st.container(border=True):
        st.markdown(f"**{user['name']}**")
        st.caption(user['role'])
        st.caption(f"ID: {user['id']}")
        st.info(f"Access: {st.session_state.current_domain}")
    
    st.divider()

    # 3. SESSION MANAGEMENT (Domain Specific)
    st.subheader("Session History")
    
    if st.session_state.current_domain not in st.session_state.domain_chats:
        st.session_state.domain_chats[st.session_state.current_domain] = {}
    
    current_domain_sessions = st.session_state.domain_chats[st.session_state.current_domain]
    session_keys = list(current_domain_sessions.keys())

    # "New Chat" Button
    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.global_session_counter += 1
        new_session_name = f"Session {st.session_state.global_session_counter}"
        
        st.session_state.domain_chats[st.session_state.current_domain][new_session_name] = []
        st.session_state.active_session_id = new_session_name
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
        
        # --- RENAME / DELETE CONTROLS ---
        with st.expander("⚙️ Manage Session"):
            # RENAME
            new_name = st.text_input("Rename Chat", value=selected_session)
            if st.button("Update Name"):
                if new_name and new_name != selected_session:
                    if new_name in current_domain_sessions:
                        st.error("Name already exists!")
                    else:
                        # Swap keys in dictionary
                        data = st.session_state.domain_chats[st.session_state.current_domain].pop(selected_session)
                        st.session_state.domain_chats[st.session_state.current_domain][new_name] = data
                        # Update active session ID
                        st.session_state.active_session_id = new_name
                        st.rerun()
            
            # DELETE
            st.markdown("---")
            if st.button("🗑️ Delete Chat", type="primary"):
                # Remove from dictionary
                del st.session_state.domain_chats[st.session_state.current_domain][selected_session]
                # Reset active session ID to force logic to pick a new one or None
                st.session_state.active_session_id = None
                st.rerun()

    else:
        st.session_state.active_session_id = None
        st.info(f"No active chats in {st.session_state.current_domain}.")

    # Logout
    st.divider()
    if st.button("Logout"):
        st.session_state.clear()
        st.rerun()

# --- MAIN CONTENT AREA ---
col_chat, col_inspector = st.columns([0.65, 0.35], gap="medium")

# --- CENTER PANEL: CHAT ---
with col_chat:
    st.subheader("HCL-Tech Enterprise Assistant")
    
    if st.session_state.active_session_id and \
       st.session_state.active_session_id in st.session_state.domain_chats[st.session_state.current_domain]:
        
        active_history = st.session_state.domain_chats[st.session_state.current_domain][st.session_state.active_session_id]
        
        chat_container = st.container(height=500, border=True)
        with chat_container:
            if not active_history:
                st.caption(f"Started new conversation in {st.session_state.current_domain}...")
            
            for message in active_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        # Input Area
        prompt = st.chat_input(f"Message {st.session_state.active_session_id}...")
        
        if prompt:
            st.session_state.domain_chats[st.session_state.current_domain][st.session_state.active_session_id].append(
                {"role": "user", "content": prompt}
            )
            
            time.sleep(0.5) 
            response_text = f"Response from {st.session_state.current_domain} agent regarding: {prompt}"
            
            st.session_state.domain_chats[st.session_state.current_domain][st.session_state.active_session_id].append(
                {"role": "assistant", "content": response_text}
            )
            st.rerun()
            
    else:
        st.info(f"👈 Click '+ New Chat' to start a {st.session_state.current_domain} session.")

# --- RIGHT PANEL: INSPECTOR ---
with col_inspector:
    st.subheader("Inspector")
    
    if st.session_state.active_session_id:
        tab1, tab2, tab3 = st.tabs(["Action JSON", "Source Preview", "Reasoning"])

        with tab1:
            st.caption("Generated Payload")
            mock_payload = {
                "tool_call": "system_action",
                "requester_id": user['id'],
                "domain_context": st.session_state.current_domain,
                "session_id": st.session_state.active_session_id,
                "timestamp": "2026-01-07T10:00:00Z"
            }
            st.json(mock_payload)

        with tab2:
            st.caption("Retrieved Context")
            st.info("📄 [PDF Preview Placeholder]")
            st.markdown(f"> *Retrieving data for {st.session_state.current_domain}...*")

        with tab3:
            with st.status("Agent Status", expanded=True):
                st.write(f"🔍 Context: {st.session_state.current_domain}")
                st.write(f"📂 Session: {st.session_state.active_session_id}")
                st.write(f"👤 User: {user['role']}")
                st.write("✅ Active")
    else:
        st.caption("Start a chat to see inspection details.")

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("© 2026 Byte Me | Kshitij NLP Challenge")