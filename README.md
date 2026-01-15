# ByteMe Enterprise Assistant

An agentic RAG-based enterprise assistant with long-term and short-term memory, built with LangGraph, PostgreSQL (Neon DB), and Redis.

## Features

### 🎯 Three Domain Tools

1. **IT Service Desk**
   - Automated troubleshooting workflows
   - Ticket creation and tracking
   - Software request management
   - Password reset assistance

2. **Developer Support**
   - Legacy code documentation retrieval
   - Code fix suggestions
   - API documentation
   - Code review checklists

3. **HR Operations**
   - Policy question answering
   - Leave application guidance
   - Benefits information
   - Onboarding checklists

### 🧠 Memory Architecture

- **Short-term Memory**: Redis-based session storage (falls back to in-memory)
- **Long-term Memory**: PostgreSQL (Neon DB) for user-specific persistent storage
- **Fallback**: ChromaDB for long-term memory if PostgreSQL unavailable
- **Auto-consolidation**: Important conversations are promoted to long-term memory

### 🔐 User Authentication

- **PostgreSQL-based**: Secure user registration and authentication via Neon DB
- **User-specific memory**: Long-term memories are stored per user ID
- **Fallback**: JSON file authentication if database unavailable

### 🔄 Agentic RAG Workflow

1. Memory Retrieval → Fetch relevant conversation history
2. Tool Detection → Identify if specific actions are needed
3. Hybrid Search → Query text and vision collections
4. Grade → Assess document relevance
5. Generate → Create grounded response
6. Reflect → Verify answer grounding
7. Store → Persist to memory

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

Required variables:
- `GROQ_API_KEY`: Your Groq API key for LLM access
- `DATABASE_URL`: Neon DB connection string (for auth & long-term memory)

Optional:
- `REDIS_HOST`, `REDIS_PORT`: Redis configuration for short-term memory

### 3. Set Up Neon DB (PostgreSQL)

1. Create a free account at [Neon.tech](https://neon.tech)
2. Create a new project and database
3. Copy the connection string to your `.env` file:

```env
DATABASE_URL=postgresql://username:password@ep-xxx.region.aws.neon.tech/dbname?sslmode=require
```

The application will automatically create the required tables on first run.

### 4. Start Redis (Optional)

If using Redis for short-term memory:

```bash
# Using Docker
docker run -d --name redis -p 6379:6379 redis

# Or install locally
# Windows: Use WSL or Redis for Windows
# Linux: sudo apt install redis-server
```

### 5. Run the Application

```bash
streamlit run app.py
```

## Project Structure

```
app/
├── app.py                 # Main Streamlit application
├── database.py            # PostgreSQL/Neon DB integration
├── engine.py              # ByteMe RAG Engine (ChromaDB, embeddings)
├── memory_manager.py      # Short-term (Redis) & Long-term (PostgreSQL) memory
├── agent.py               # LangGraph agentic workflow
├── tools/
│   ├── __init__.py
│   ├── it_service_desk.py # IT support automation
│   ├── developer_support.py # Developer tools
│   └── hr_operations.py   # HR automation
├── cred.json              # Fallback user credentials (JSON)
├── chroma_db/             # ChromaDB persistent storage (fallback)
├── requirements.txt       # Python dependencies
└── .env.example           # Environment variables template
```

## Database Schema

The PostgreSQL database includes:

### Users Table
- `user_id`: Unique employee ID
- `username`: Login username
- `password_hash`: SHA-256 hashed password
- `name`, `email`, `role`: User profile
- `allowed_domains`: Array of accessible domains

### Long-term Memory Table
- `memory_id`: Unique memory identifier
- `user_id`: Owner of the memory
- `question`, `answer`: Conversation content
- `domain`: Domain context
- `embedding`: Vector embedding for semantic search
- `importance_score`: Memory importance rating

## Demo Credentials (JSON Fallback)

| Username | Password    | Role           | Domains |
|----------|-------------|----------------|---------|
| john     | password123 | IT-Admin       | IT Service Desk, Developer Support |
| sarah    | hrpass      | HR Manager     | HR Operations |
| mike     | devpass     | Lead Developer | Developer Support, IT Service Desk |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Frontend                        │
│                  (Sign In / Sign Up)                        │
├─────────────────────────────────────────────────────────────┤
│                    LangGraph Agent                           │
│  ┌─────────┐  ┌──────────┐  ┌─────────┐  ┌───────────────┐ │
│  │ Memory  │→ │ Retrieve │→ │ Grade   │→ │   Generate    │ │
│  │Retrieval│  │ (Hybrid) │  │ (LLM)   │  │   (RAG LLM)   │ │
│  └─────────┘  └──────────┘  └─────────┘  └───────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐     ┌────────────────────────────────┐ │
│  │  Short-term     │     │      Long-term Memory          │ │
│  │  Memory (Redis) │     │      (PostgreSQL/Neon DB)      │ │
│  │  - Session data │     │  - User authentication         │ │
│  │  - Recent QA    │     │  - Per-user conversation       │ │
│  └─────────────────┘     │    history with embeddings     │ │
│                          └────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                    Domain Tools                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ IT Service   │  │  Developer   │  │  HR Operations   │  │
│  │ Desk Tool    │  │ Support Tool │  │     Tool         │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Usage Examples

### User Registration

```
1. Open the app and go to "Sign Up" tab
2. Fill in username, password, name, email
3. Select your role and access domains
4. Click "Create Account"
5. Sign in with your new credentials
```

### IT Service Desk

```
User: "I can't connect to the VPN"
Assistant: Here are troubleshooting steps...
           Would you like me to create a ticket?

User: "Yes, create a ticket for VPN connection issue"
Assistant: ✅ Ticket INC20260116ABC123 created.
           Priority: Medium, SLA: 24 hours
```

### Developer Support

```
User: "Explain the authentication module"
Assistant: The auth_module handles user authentication...
           Functions: authenticate_user(), verify_token()

User: "How to fix a null pointer error?"
Assistant: Add null checks before accessing properties...
           [Code example provided]
```

### HR Operations

```
User: "What's the leave policy for annual leave?"
Assistant: Annual Leave Policy:
           - Entitlement: 20 days/year
           - Carryover: Max 5 days to next year

User: "Apply for leave from Jan 20 to Jan 25"
Assistant: ✅ Leave request LV20260116XYZ submitted.
           Status: Pending Approval
```

## License

© 2026 Byte Me | Kshitij NLP Challenge
