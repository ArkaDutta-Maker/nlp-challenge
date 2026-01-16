"""
Agent - LangGraph-based Agentic RAG Workflow
Based on the notebook's agentic architecture with memory augmentation
"""

import json
import ast
import os
from typing import List, TypedDict, Dict, Any
from datetime import datetime

# LangChain / LangGraph imports
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END

# Import tools
from tools.it_service_desk import ITServiceDeskTool
from tools.developer_support import DeveloperSupportTool
from tools.hr_operations import HROperationsTool
from tools.web_search import extract_and_search_hyperlinks, web_search_action


# --- HELPER FUNCTIONS ---

def parse_json_safe(text_output: str) -> Dict:
    """
    Safely parses LLM output that might use single quotes or markdown blocks.
    """
    text = text_output.strip().replace("```json", "").replace("```", "")
    
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(text)
        except:
            if "yes" in text.lower():
                return {"score": "yes"}
            return {"score": "no"}


# --- AGENT STATE ---

class AgentState(TypedDict):
    """State for the agentic RAG workflow"""
    question: str
    original_question: str
    domain: str
    user_id: str
    session_id: str
    documents: List[str]
    document_sources: List[str]  # Track document sources
    retrieval_results: List[Dict]  # Full retrieval results for preview
    generation: str
    is_grounded: bool
    is_memory_question: bool  # Whether this is a memory/conversation question
    is_page_question: bool  # Whether this is a page-specific question
    page_number: int  # Extracted page number for page queries
    retries: int
    memory_context: str
    long_term_memory: str
    should_store_memory: bool
    tool_calls: List[Dict]  # For tracking tool usage
    tool_result: Dict  # Result from tool execution
    reasoning_steps: List[str]  # For inspector panel
    web_search_results: str  # Results from web search


class ByteMeAgent:
    """
    Memory-Augmented Agentic RAG Agent using LangGraph.
    """
    
    def __init__(self, engine, memory_manager, groq_api_key: str = None):
        """
        Initialize the agent.
        
        Args:
            engine: ByteMeEngine instance for retrieval
            memory_manager: MemoryManager instance
            groq_api_key: Groq API key (or set GROQ_API_KEY env var)
        """
        self.engine = engine
        self.memory_manager = memory_manager
        
        # Initialize domain tools
        self.tools = {
            "IT Service Desk": ITServiceDeskTool(),
            "Developer Support": DeveloperSupportTool(),
            "HR Operations": HROperationsTool()
        }
        
        # Set API key
        if groq_api_key:
            os.environ["GROQ_API_KEY"] = groq_api_key
        
        # Initialize LLMs
        self.llm_router = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
        self.llm_gen = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)
        
        # Build chains
        self._build_chains()
        
        # Build workflow graph
        self.app = self._build_graph()
        
        print("✅ ByteMeAgent initialized with LangGraph workflow")
    
    def _build_chains(self):
        """Build LangChain chains for various tasks"""
        
        # Domain-specific system prompts
        self.domain_prompts = {
            "IT Service Desk": """You are a CONFIDENT and AUTHORITATIVE IT Service Desk assistant. 

COMMUNICATION STYLE:
- NEVER use hedging language like "it appears", "it seems", "I think", "probably", "might be", "could be", "possibly"
- ALWAYS speak with certainty: "This is...", "The solution is...", "You need to...", "The issue is..."
- State facts directly and provide clear, actionable guidance

You help with:
- Troubleshooting technical issues
- Creating support tickets  
- Software installation requests
- Password resets and access issues
- Network and connectivity problems

Be professional, follow ITIL best practices, and offer to create a ticket if the issue cannot be resolved immediately.""",
            
            "Developer Support": """You are a CONFIDENT and EXPERT Developer Support assistant.

COMMUNICATION STYLE:
- NEVER use hedging language like "it appears", "it seems", "I think", "probably", "might be", "could be"
- ALWAYS speak with technical authority: "The code does...", "This function...", "The solution is...", "You should..."
- State technical facts directly and confidently

You help with:
- Explaining legacy code and documentation
- Suggesting code fixes and improvements
- Debugging assistance
- API documentation and usage
- Best practices and code review

Provide code examples when helpful and explain technical concepts with confidence and clarity.""",
            
            "HR Operations": """You are a CONFIDENT and KNOWLEDGEABLE HR Operations assistant.

COMMUNICATION STYLE:
- NEVER use hedging language like "it appears", "it seems", "I believe", "probably", "might be"
- ALWAYS speak with clarity: "The policy states...", "You are entitled to...", "The process is...", "According to company guidelines..."
- State HR information and policies directly and confidently

You help with:
- Company policy questions
- Leave application guidance
- Benefits information
- Onboarding procedures
- Performance review processes

Be professional, maintain confidentiality, and direct sensitive matters to HR personnel when appropriate."""
        }
        
        # RAG Chain with Memory Context
        self.rag_prompt = PromptTemplate(
            template="""You are a confident, knowledgeable enterprise assistant with access to document context and conversation history.

{domain_system_prompt}

📚 DOCUMENT CONTEXT:
{context}

🧠 RECENT CONVERSATION:
{memory_context}

🧠 RELEVANT PAST CONVERSATIONS:
{long_term_memory}

❓ CURRENT QUESTION: {question}

CRITICAL INSTRUCTIONS:
- Use the document context as your primary source of truth
- Answer with CONFIDENCE and AUTHORITY - state facts directly without hedging
- NEVER use phrases like "it appears", "it seems", "I think", "probably", "might be", "could be", "possibly"
- ALWAYS use definitive language: "The document states...", "According to the report...", "This is...", "The answer is..."
- Reference conversation history to maintain context continuity
- Be concise, accurate, and professional
- If information is genuinely not in the documents, clearly state: "This information is not present in the available documents."

Provide a direct, confident answer:""",
            input_variables=["domain_system_prompt", "context", "memory_context", "long_term_memory", "question"]
        )
        self.rag_chain = self.rag_prompt | self.llm_gen | StrOutputParser()
        
        # Rewriter Chain
        self.rewrite_prompt = PromptTemplate(
            template="""You are a search query optimizer. 
Rewrite the user's question to be short, specific, and keyword-rich for a vector database.
Do NOT output a list. Output ONLY the rewritten query string.

Original: {question}
New Query:""",
            input_variables=["question"]
        )
        self.rewriter_chain = self.rewrite_prompt | self.llm_router | StrOutputParser()
        
        # Grader Chain
        self.grade_prompt = PromptTemplate(
            template="""You are a grader assessing relevance. 
Does the document contain ANY information related to the user's question?

Document: {document}
Question: {question}

Return strictly valid JSON: {{"score": "yes"}} or {{"score": "no"}}.""",
            input_variables=["question", "document"]
        )
        self.grader_chain = self.grade_prompt | self.llm_router | StrOutputParser()
        
        # Grounding Chain
        self.grounding_prompt = PromptTemplate(
            template="""Context: {context} 
Answer: {generation} 

Is the answer fully supported by the context? 
Return strictly valid JSON: {{"score": "yes"}} or {{"score": "no"}}.""",
            input_variables=["context", "generation"]
        )
        self.grounding_chain = self.grounding_prompt | self.llm_router | StrOutputParser()
        
        # Tool Detection Chain - Enhanced for all tool actions
        self.tool_prompt = PromptTemplate(
            template="""Analyze the user's question and determine if a specific action/tool is needed.

Domain: {domain}
Question: {question}
Conversation Context: {context}

== IT SERVICE DESK ACTIONS ==
- "create_ticket": User wants to create/open a support ticket, report an issue
  Parameters: issue (string), category (network/email/software/hardware/access/security), priority (low/medium/high/critical), description (string)
  
- "check_status": User wants to check ticket status, view their tickets
  Parameters: ticket_id (string, optional), user_id (string)
  
- "password_reset": User needs password reset, locked out, forgot password
  Parameters: target_system (AD/email/vpn/application), reason (string)
  
- "software_request": User needs software/application installed
  Parameters: software_name (string), justification (string)
  
- "troubleshoot": User has a technical issue and needs help troubleshooting
  Parameters: category (network/email/software/hardware/printer/vpn/password), symptoms (string)
  
- "system_status": User wants to know if systems are working
  Parameters: system (string, optional - email_server/vpn/active_directory/file_server/intranet)
  
- "escalate": User wants to escalate an existing ticket
  Parameters: ticket_id (string), reason (string)
  
- "knowledge": User wants to search knowledge base or get help article
  Parameters: topic (string)

== DEVELOPER SUPPORT ACTIONS ==
- "code_explanation": User wants code explained
  Parameters: code (string), language (string)
  
- "suggest_fix": User wants bug fix suggestions
  Parameters: code (string), error (string)
  
- "api_docs": User needs API documentation
  Parameters: endpoint (string), method (string)
  
- "code_review": User wants code reviewed
  Parameters: code (string), language (string)

== HR OPERATIONS ACTIONS ==
- "leave_application": User wants to apply for leave
  Parameters: leave_type (string), start_date (string), end_date (string), reason (string)
  
- "policy_query": User asking about company policies
  Parameters: policy_type (string)
  
- "benefits_info": User asking about employee benefits
  Parameters: benefit_type (string)
  
- "payroll_query": User asking about salary/payroll
  Parameters: query_type (string)

IMPORTANT: 
- Return {{"tool": "none"}} if this is a general question that should use document retrieval
- Extract relevant parameters from the user's question
- If the user is describing an IT issue, use "troubleshoot" action first to provide immediate help

Return strictly valid JSON with "tool" and "parameters" fields.
Examples:
- {{"tool": "troubleshoot", "parameters": {{"category": "network", "symptoms": "cannot connect to wifi"}}}}
- {{"tool": "create_ticket", "parameters": {{"issue": "Laptop not starting", "category": "hardware", "priority": "high"}}}}
- {{"tool": "none"}}

Response:""",
            input_variables=["domain", "question", "context"]
        )
        self.tool_chain = self.tool_prompt | self.llm_router | StrOutputParser()
        
        # Memory-based answer chain (for conversation summary, recall, etc.)
        self.memory_answer_prompt = PromptTemplate(
            template="""{domain_system_prompt}

The user is asking about your previous conversation or wants you to recall/summarize what was discussed.

🧠 RECENT CONVERSATION HISTORY:
{memory_context}

🧠 RELEVANT PAST CONVERSATIONS:
{long_term_memory}

❓ USER'S REQUEST: {question}

CRITICAL INSTRUCTIONS:
- Answer based ONLY on the conversation history provided above
- Speak with CONFIDENCE - state what was discussed directly
- NEVER use hedging phrases like "it appears", "it seems", "I believe", "probably"
- Use definitive language: "We discussed...", "You asked about...", "I explained that..."
- If asking for a summary, provide a clear, concise summary of key points
- If conversation history is empty, state clearly: "We haven't discussed this topic yet."

Direct answer:""",
            input_variables=["domain_system_prompt", "memory_context", "long_term_memory", "question"]
        )
        self.memory_answer_chain = self.memory_answer_prompt | self.llm_gen | StrOutputParser()
        
        # Page-based answer chain
        self.page_answer_prompt = PromptTemplate(
            template="""{domain_system_prompt}

The user is asking about content from a specific page in the document.

📄 PAGE {page_number} CONTENT:
{page_content}

🧠 CONVERSATION CONTEXT:
{memory_context}

❓ USER'S QUESTION: {question}

CRITICAL INSTRUCTIONS:
- Answer with CONFIDENCE based on the page content provided
- NEVER use hedging phrases like "it appears", "it seems", "I think", "probably", "might be"
- Use DEFINITIVE language: "Page {page_number} shows...", "On this page...", "The content states..."
- Reference the page number directly in your response
- State facts as facts - you have the actual page content
- If specific information is not on this page, state clearly: "This information is not present on page {page_number}."

Direct, confident answer:""",
            input_variables=["domain_system_prompt", "page_number", "page_content", "memory_context", "question"]
        )
        self.page_answer_chain = self.page_answer_prompt | self.llm_gen | StrOutputParser()
    
    def _is_memory_question(self, question: str) -> bool:
        """Check if the question is about conversation history/memory"""
        memory_keywords = [
            "summarize", "summary", "what did we", "what have we", 
            "discussed", "talked about", "conversation so far",
            "recap", "previous", "earlier", "remember when",
            "you said", "i said", "we discussed", "our conversation",
            "what was", "remind me", "go over", "review",
            "so far", "up to now", "until now", "thus far"
        ]
        question_lower = question.lower()
        return any(keyword in question_lower for keyword in memory_keywords)
    
    def _extract_page_number(self, question: str) -> int:
        """Extract page number from question if present"""
        import re
        question_lower = question.lower()
        
        # Patterns to match page references
        patterns = [
            r'page\s*(?:number\s*)?(\d+)',  # "page 5", "page number 5"
            r'pg\.?\s*(\d+)',                # "pg 5", "pg. 5"
            r'p\.?\s*(\d+)',                 # "p 5", "p. 5"
            r'on\s+(\d+)(?:st|nd|rd|th)?\s+page',  # "on 5th page"
            r'(\d+)(?:st|nd|rd|th)?\s+page',  # "5th page"
            r'from\s+page\s*(\d+)',          # "from page 5"
            r'in\s+page\s*(\d+)',            # "in page 5"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, question_lower)
            if match:
                return int(match.group(1))
        
        return None
    
    def _is_page_question(self, question: str) -> bool:
        """Check if the question is asking about a specific page"""
        return self._extract_page_number(question) is not None
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        
        # Define nodes
        def memory_retrieval_node(state: AgentState) -> Dict:
            """Retrieve relevant memories"""
            reasoning = ["🧠 Retrieving relevant memories..."]
            
            try:
                context = self.memory_manager.get_context(
                    session_id=state["session_id"],
                    user_id=state["user_id"],
                    query=state["question"],
                    domain=state["domain"]
                )
                
                reasoning.append(f"   - Short-term: {len(context.get('short_term', ''))} chars")
                reasoning.append(f"   - Long-term: {len(context.get('long_term', ''))} chars")
                
                return {
                    "memory_context": context.get("short_term", ""),
                    "long_term_memory": context.get("long_term", ""),
                    "original_question": state.get("original_question") or state["question"],
                    "reasoning_steps": state.get("reasoning_steps", []) + reasoning
                }
            except Exception as e:
                reasoning.append(f"   - ⚠️ Memory retrieval error: {str(e)}")
                return {
                    "memory_context": "",
                    "long_term_memory": "",
                    "original_question": state.get("original_question") or state["question"],
                    "reasoning_steps": state.get("reasoning_steps", []) + reasoning
                }
        
        def check_memory_question_node(state: AgentState) -> Dict:
            """Check if this is a memory/conversation-related question or page-specific question"""
            reasoning = ["🔎 Checking question type..."]
            
            is_memory_q = self._is_memory_question(state["question"])
            is_page_q = self._is_page_question(state["question"])
            page_num = self._extract_page_number(state["question"]) if is_page_q else None
            
            if is_memory_q:
                reasoning.append("   - ✓ This is a memory/conversation question")
                reasoning.append("   - Will answer from conversation history")
            elif is_page_q:
                reasoning.append(f"   - ✓ This is a page-specific question (Page {page_num})")
                reasoning.append("   - Will retrieve content from specified page")
            else:
                reasoning.append("   - This requires document retrieval")
            
            return {
                "is_memory_question": is_memory_q,
                "is_page_question": is_page_q,
                "page_number": page_num,
                "reasoning_steps": state.get("reasoning_steps", []) + reasoning
            }
        
        def page_retrieve_node(state: AgentState) -> Dict:
            """Retrieve content from a specific page"""
            page_num = state.get("page_number", 1)
            reasoning = [f"📄 Retrieving content from Page {page_num}..."]
            
            try:
                # Get page content from engine
                page_results = self.engine.get_page_content(page_num)
                
                if page_results:
                    docs = [r["content"] for r in page_results]
                    sources = [f"{r.get('source', 'Document')} (Page {page_num})" for r in page_results]
                    reasoning.append(f"   - Found {len(docs)} chunks from page {page_num}")
                    
                    return {
                        "documents": docs,
                        "document_sources": sources,
                        "retrieval_results": page_results,
                        "reasoning_steps": state.get("reasoning_steps", []) + reasoning
                    }
                else:
                    # Fallback: search with page filter
                    search_results = self.engine.search_by_page(state["question"], page_num, k=5)
                    if search_results:
                        docs = [r["content"] for r in search_results]
                        sources = [r.get("source", f"Page {page_num}") for r in search_results]
                        reasoning.append(f"   - Found {len(docs)} relevant results from page {page_num}")
                        
                        return {
                            "documents": docs,
                            "document_sources": sources,
                            "retrieval_results": search_results,
                            "reasoning_steps": state.get("reasoning_steps", []) + reasoning
                        }
                    
                    reasoning.append(f"   - ⚠️ No content found for page {page_num}")
                    # Get available pages to inform user
                    available = self.engine.get_available_pages()
                    if available:
                        reasoning.append(f"   - Available pages: {available[:10]}{'...' if len(available) > 10 else ''}")
                    
                    return {
                        "documents": [],
                        "document_sources": [],
                        "retrieval_results": [],
                        "reasoning_steps": state.get("reasoning_steps", []) + reasoning
                    }
            except Exception as e:
                reasoning.append(f"   - ❌ Page retrieval error: {str(e)}")
                return {
                    "documents": [],
                    "document_sources": [],
                    "retrieval_results": [],
                    "reasoning_steps": state.get("reasoning_steps", []) + reasoning
                }
        
        def page_answer_node(state: AgentState) -> Dict:
            """Generate answer based on page-specific content"""
            page_num = state.get("page_number", 1)
            reasoning = [f"💡 Generating answer from Page {page_num} content..."]
            
            try:
                domain_prompt = self.domain_prompts.get(
                    state["domain"],
                    "You are a helpful enterprise assistant."
                )
                
                page_content = "\n\n".join(state.get("documents", [])) if state.get("documents") else f"No content found for page {page_num}."
                
                gen = self.page_answer_chain.invoke({
                    "domain_system_prompt": domain_prompt,
                    "page_number": page_num,
                    "page_content": page_content,
                    "memory_context": state.get("memory_context", ""),
                    "question": state["question"]
                })
                
                reasoning.append(f"   - Generated {len(gen)} chars")
                
                return {
                    "generation": gen,
                    "is_grounded": True,  # Page-specific answers are grounded by definition
                    "should_store_memory": True,
                    "reasoning_steps": state.get("reasoning_steps", []) + reasoning
                }
            except Exception as e:
                reasoning.append(f"   - ❌ Generation error: {str(e)}")
                return {
                    "generation": f"I couldn't find content for page {page_num}. Please check the page number and try again.",
                    "is_grounded": False,
                    "should_store_memory": False,
                    "reasoning_steps": state.get("reasoning_steps", []) + reasoning
                }
        
        def memory_answer_node(state: AgentState) -> Dict:
            """Generate answer directly from memory for conversation-related questions"""
            reasoning = ["💭 Answering from conversation memory..."]
            
            try:
                domain_prompt = self.domain_prompts.get(
                    state["domain"],
                    "You are a helpful enterprise assistant."
                )
                
                memory_context = state.get("memory_context", "")
                long_term = state.get("long_term_memory", "")
                
                # Check if we have any memory to work with
                if not memory_context and not long_term:
                    reasoning.append("   - No conversation history found")
                    return {
                        "generation": "I don't have any previous conversation history to summarize. This appears to be the start of our conversation. How can I help you today?",
                        "is_grounded": True,
                        "should_store_memory": False,
                        "reasoning_steps": state.get("reasoning_steps", []) + reasoning
                    }
                
                gen = self.memory_answer_chain.invoke({
                    "domain_system_prompt": domain_prompt,
                    "memory_context": memory_context or "No recent conversation.",
                    "long_term_memory": long_term or "No past conversations.",
                    "question": state["question"]
                })
                
                reasoning.append(f"   - Generated {len(gen)} chars from memory")
                
                return {
                    "generation": gen,
                    "is_grounded": True,  # Memory answers are inherently "grounded" in conversation
                    "should_store_memory": True,
                    "reasoning_steps": state.get("reasoning_steps", []) + reasoning
                }
            except Exception as e:
                reasoning.append(f"   - ❌ Error: {str(e)}")
                return {
                    "generation": "I encountered an error while summarizing our conversation. Please try again.",
                    "is_grounded": False,
                    "should_store_memory": False,
                    "reasoning_steps": state.get("reasoning_steps", []) + reasoning
                }
        
        def tool_detection_node(state: AgentState) -> Dict:
            """Detect if a specific tool/action is needed"""
            reasoning = ["🔧 Detecting required tools..."]
            
            # Build context from memory and documents
            context_parts = []
            if state.get("memory_context"):
                context_parts.append(f"Recent conversation: {state['memory_context'][:200]}")
            if state.get("documents"):
                context_parts.append(f"Retrieved info: {' '.join(state['documents'])[:200]}")
            context = " | ".join(context_parts) if context_parts else "No prior context"
            
            try:
                raw_res = self.tool_chain.invoke({
                    "domain": state["domain"],
                    "question": state["question"],
                    "context": context
                })
                tool_info = parse_json_safe(raw_res)
                
                if tool_info.get("tool") and tool_info["tool"] != "none":
                    reasoning.append(f"   - Detected tool: {tool_info['tool']}")
                    reasoning.append(f"   - Parameters: {tool_info.get('parameters', {})}")
                    return {
                        "tool_calls": [tool_info],
                        "reasoning_steps": state.get("reasoning_steps", []) + reasoning
                    }
            except Exception as e:
                reasoning.append(f"   - Tool detection error: {str(e)}")
            
            reasoning.append("   - No specific tool needed")
            return {
                "tool_calls": [],
                "reasoning_steps": state.get("reasoning_steps", []) + reasoning
            }
        
        def retrieve_node(state: AgentState) -> Dict:
            """Hybrid retrieval from knowledge base"""
            reasoning = [f"🔍 Searching for: '{state['question'][:50]}...'"]
            
            try:
                results = self.engine.hybrid_search(
                    query=state["question"],
                    domain=state["domain"],
                    k=5
                )
                
                # Extract content and preserve metadata
                docs = []
                doc_sources = []
                for r in results:
                    docs.append(r["content"])
                    source_info = f"{r.get('source', 'Unknown')} ({r.get('type', 'text')})"
                    doc_sources.append(source_info)
                    
                reasoning.append(f"   - Found {len(docs)} documents")
                if doc_sources:
                    reasoning.append(f"   - Sources: {', '.join(doc_sources[:3])}")
                
                return {
                    "documents": docs,
                    "document_sources": doc_sources,
                    "retrieval_results": results,  # Store full results for preview
                    "retries": state.get("retries", 0),
                    "reasoning_steps": state.get("reasoning_steps", []) + reasoning
                }
            except Exception as e:
                reasoning.append(f"   - ❌ Retrieval error: {str(e)}")
                return {
                    "documents": [],
                    "document_sources": [],
                    "retrieval_results": [],
                    "retries": state.get("retries", 0) + 1,
                    "reasoning_steps": state.get("reasoning_steps", []) + reasoning
                }
        
        def grade_node(state: AgentState) -> Dict:
            """Grade retrieved documents for relevance"""
            reasoning = ["📝 Grading document relevance..."]
            relevant = []
            relevant_results = []
            
            if not state.get("documents"):
                reasoning.append("   - No documents to grade")
                return {
                    "documents": [],
                    "retrieval_results": [],
                    "retries": state.get("retries", 0) + 1,
                    "reasoning_steps": state.get("reasoning_steps", []) + reasoning
                }
            
            retrieval_results = state.get("retrieval_results", [])
            
            for i, doc in enumerate(state["documents"]):
                try:
                    raw_res = self.grader_chain.invoke({
                        "question": state["question"],
                        "document": doc[:500]
                    })
                    parsed_res = parse_json_safe(raw_res)
                    
                    if parsed_res.get("score") == "yes":
                        relevant.append(doc)
                        # Preserve corresponding retrieval result
                        if i < len(retrieval_results):
                            relevant_results.append(retrieval_results[i])
                        reasoning.append(f"   - Doc {i+1}: ✓ Relevant")
                    else:
                        reasoning.append(f"   - Doc {i+1}: ✗ Not relevant")
                except Exception as e:
                    reasoning.append(f"   - Doc {i+1}: ⚠️ Grade error: {str(e)[:50]}")
                    # Keep document on error to be safe
                    relevant.append(doc)
                    if i < len(retrieval_results):
                        relevant_results.append(retrieval_results[i])
            
            new_retries = state.get("retries", 0) + 1 if not relevant else state.get("retries", 0)
            reasoning.append(f"   - {len(relevant)}/{len(state['documents'])} documents passed")
            
            return {
                "documents": relevant,
                "retrieval_results": relevant_results,
                "retries": new_retries,
                "reasoning_steps": state.get("reasoning_steps", []) + reasoning
            }
        
        def rewrite_node(state: AgentState) -> Dict:
            """Rewrite query for better retrieval"""
            reasoning = ["🔄 Rewriting query for better results..."]
            
            try:
                new_q = self.rewriter_chain.invoke({"question": state["question"]})
                new_q = new_q.strip().replace('"', '')
                reasoning.append(f"   - New query: '{new_q[:50]}...'")
            except Exception as e:
                new_q = state["question"]
                reasoning.append(f"   - ⚠️ Rewrite error, using original: {str(e)[:50]}")
            
            return {
                "question": new_q,
                "reasoning_steps": state.get("reasoning_steps", []) + reasoning
            }
        
        def tool_execution_node(state: AgentState) -> Dict:
            """Execute detected tools and return results"""
            reasoning = ["⚙️ Executing tool action..."]
            
            tool_calls = state.get("tool_calls", [])
            if not tool_calls:
                reasoning.append("   - No tool actions to execute")
                return {
                    "tool_result": {},
                    "reasoning_steps": state.get("reasoning_steps", []) + reasoning
                }
            
            tool_call = tool_calls[0]  # Execute first tool
            action = tool_call.get("tool", "none")
            parameters = tool_call.get("parameters", {})
            
            # Add user context to parameters
            parameters["user_id"] = state.get("user_id", "user")
            
            # Add retrieved document context if available
            if state.get("documents"):
                context = "\n".join(state["documents"][:2])  # First 2 relevant docs
                parameters["context"] = context
            
            # Get the appropriate tool based on domain
            domain = state.get("domain", "IT Service Desk")
            tool = self.tools.get(domain)
            
            if not tool:
                reasoning.append(f"   - ⚠️ No tool found for domain: {domain}")
                return {
                    "tool_result": {"error": f"No tool available for {domain}"},
                    "reasoning_steps": state.get("reasoning_steps", []) + reasoning
                }
            
            try:
                # Execute the action
                result = tool.execute_action(action, parameters)
                reasoning.append(f"   - ✓ Executed: {action}")
                reasoning.append(f"   - Result: {'Success' if result.get('success') else 'Failed'}")
                
                return {
                    "tool_result": result,
                    "reasoning_steps": state.get("reasoning_steps", []) + reasoning
                }
            except Exception as e:
                reasoning.append(f"   - ❌ Execution error: {str(e)}")
                return {
                    "tool_result": {"error": str(e), "success": False},
                    "reasoning_steps": state.get("reasoning_steps", []) + reasoning
                }
        
        def generate_node(state: AgentState) -> Dict:
            """Generate answer using RAG and tool results"""
            reasoning = ["💡 Generating answer..."]
            
            try:
                domain_prompt = self.domain_prompts.get(
                    state["domain"],
                    "You are a helpful enterprise assistant."
                )
                
                # Build context from documents
                doc_context = "\n\n".join(state.get("documents", [])) if state.get("documents") else ""
                
                # Include tool result if available
                tool_result = state.get("tool_result", {})
                tool_context = ""
                if tool_result and tool_result.get("success"):
                    # Format tool result for inclusion in context
                    tool_context = f"\n\n== TOOL ACTION RESULT ==\n{json.dumps(tool_result, indent=2, default=str)}"
                    reasoning.append("   - Including tool result in context")
                
                # Combine contexts
                if doc_context and tool_context:
                    context = f"📚 Retrieved Information:\n{doc_context}{tool_context}"
                elif tool_context:
                    context = f"⚙️ Action Result:{tool_context}"
                elif doc_context:
                    context = doc_context
                else:
                    context = "No relevant documents found."
                
                gen = self.rag_chain.invoke({
                    "domain_system_prompt": domain_prompt,
                    "context": context,
                    "question": state["question"],
                    "memory_context": state.get("memory_context", ""),
                    "long_term_memory": state.get("long_term_memory", "")
                })
                
                reasoning.append(f"   - Generated {len(gen)} chars")
                
                return {
                    "generation": gen,
                    "should_store_memory": True,
                    "reasoning_steps": state.get("reasoning_steps", []) + reasoning
                }
            except Exception as e:
                reasoning.append(f"   - ❌ Generation error: {str(e)}")
                return {
                    "generation": f"I encountered an error while generating a response. Error: {str(e)[:100]}",
                    "should_store_memory": False,
                    "reasoning_steps": state.get("reasoning_steps", []) + reasoning
                }
        
        def graceful_fail_node(state: AgentState) -> Dict:
            """Handle max retries"""
            reasoning = ["❌ Max retries reached"]
            
            return {
                "generation": "The requested information is not available in the current documents. Please provide more specific details or try a different query.",
                "should_store_memory": False,
                "is_grounded": False,
                "reasoning_steps": state.get("reasoning_steps", []) + reasoning
            }
        
        def reflection_node(state: AgentState) -> Dict:
            """Verify answer grounding"""
            reasoning = ["🛡️ Verifying answer grounding..."]
            
            if "not available" in state["generation"].lower() or "not present" in state["generation"].lower():
                reasoning.append("   - Information not found response detected")
                return {
                    "is_grounded": False,
                    "reasoning_steps": state.get("reasoning_steps", []) + reasoning
                }
            
            try:
                raw_res = self.grounding_chain.invoke({
                    "context": "\n\n".join(state["documents"]),
                    "generation": state["generation"]
                })
                parsed_res = parse_json_safe(raw_res)
                
                if parsed_res.get("score") == "yes":
                    reasoning.append("   - ✓ Answer is grounded")
                    return {
                        "is_grounded": True,
                        "reasoning_steps": state.get("reasoning_steps", []) + reasoning
                    }
            except:
                pass
            
            reasoning.append("   - ⚠ Additional verification needed")
            return {
                "is_grounded": False,
                "generation": state["generation"],
                "reasoning_steps": state.get("reasoning_steps", []) + reasoning
            }
        
        def web_search_node(state: AgentState) -> Dict:
            """Search web content from hyperlinks found in the answer and enhance the response."""
            reasoning = ["🌐 Searching for hyperlinks in answer..."]
            
            try:
                current_answer = state.get("generation", "")
                reasoning.append(f"   - Analyzing answer (length: {len(current_answer)})")
                
                if current_answer:
                    # Extract and fetch web content
                    web_content = extract_and_search_hyperlinks(current_answer)
                    
                    if web_content.strip():
                        reasoning.append("   - ✓ Found hyperlinks, fetched web content")
                        reasoning.append(f"   - Web content length: {len(web_content)}")
                        
                        # Re-generate an enhanced answer that incorporates web content
                        try:
                            enhance_prompt = PromptTemplate(
                                template="""You are a confident, authoritative assistant. The user asked a question and you have both document information AND live content from the referenced URL.

**User Question:** {question}

**Initial Answer (from documents):** {initial_answer}

**Live Content Retrieved from the URL:**
{web_content}

Provide an **enhanced, confident answer** following these CRITICAL rules:
1. State facts DIRECTLY and CONFIDENTLY - never hedge
2. NEVER use phrases like "it appears", "it seems", "I think", "probably", "might be", "could be"
3. USE definitive language: "The link leads to...", "The webpage contains...", "This page offers...", "You will find..."
4. Integrate the URL content naturally into your answer
5. Tell the user exactly what they will see when they visit the link
6. Be well-formatted and professional

Confident, enhanced answer:""",
                                input_variables=["question", "initial_answer", "web_content"]
                            )
                            
                            enhance_chain = enhance_prompt | self.llm_gen | StrOutputParser()
                            
                            enhanced_answer = enhance_chain.invoke({
                                "question": state.get("original_question", state.get("question", "")),
                                "initial_answer": current_answer,
                                "web_content": web_content
                            })
                            
                            reasoning.append("   - ✓ Enhanced answer with web content")
                            
                            return {
                                "generation": enhanced_answer,
                                "web_search_results": web_content,
                                "reasoning_steps": state.get("reasoning_steps", []) + reasoning
                            }
                        except Exception as enhance_error:
                            reasoning.append(f"   - ⚠ Enhancement failed, appending web content: {str(enhance_error)}")
                            # Fallback: just append web content
                            enhanced_answer = current_answer + "\n\n---\n**📌 Additional Context from the Link:**\n" + web_content
                            return {
                                "generation": enhanced_answer,
                                "web_search_results": web_content,
                                "reasoning_steps": state.get("reasoning_steps", []) + reasoning
                            }
                    else:
                        reasoning.append("   - No hyperlinks found or web content unavailable")
                        return {
                            "web_search_results": "",
                            "reasoning_steps": state.get("reasoning_steps", []) + reasoning
                        }
                else:
                    reasoning.append("   - No answer to search for hyperlinks")
                    return {
                        "web_search_results": "",
                        "reasoning_steps": state.get("reasoning_steps", []) + reasoning
                    }
                    
            except Exception as e:
                reasoning.append(f"   - ❌ Web search failed: {str(e)}")
                print(f"Web search error details: {e}")  # Additional console logging
                return {
                    "web_search_results": "",
                    "reasoning_steps": state.get("reasoning_steps", []) + reasoning
                }
        
        def memory_storage_node(state: AgentState) -> Dict:
            """Store successful exchange to memory"""
            reasoning = ["💾 Storing to memory..."]
            
            if state.get("should_store_memory"):
                # Determine importance based on grounding and length
                importance = 0.7 if state.get("is_grounded") else 0.4
                
                self.memory_manager.add_exchange(
                    session_id=state["session_id"],
                    user_id=state["user_id"],
                    question=state["original_question"],
                    answer=state["generation"],
                    domain=state["domain"],
                    store_long_term=state.get("is_grounded", False),
                    importance=importance
                )
                reasoning.append("   - ✓ Memory stored")
            else:
                reasoning.append("   - Skipped (not storing)")
            
            return {
                "reasoning_steps": state.get("reasoning_steps", []) + reasoning
            }
        
        # Build graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("memory_retrieval", memory_retrieval_node)
        workflow.add_node("check_memory_question", check_memory_question_node)
        workflow.add_node("memory_answer", memory_answer_node)
        workflow.add_node("page_retrieve", page_retrieve_node)  # Page-specific retrieval
        workflow.add_node("page_answer", page_answer_node)  # Page-specific answer
        workflow.add_node("tool_detection", tool_detection_node)
        workflow.add_node("retrieve", retrieve_node)
        workflow.add_node("grade", grade_node)
        workflow.add_node("rewrite", rewrite_node)
        workflow.add_node("generate", generate_node)
        workflow.add_node("reflect", reflection_node)
        workflow.add_node("web_search", web_search_node)
        workflow.add_node("memory_storage", memory_storage_node)
        workflow.add_node("fail", graceful_fail_node)
        
        # Set entry point
        workflow.set_entry_point("memory_retrieval")
        
        # Add edges - First check if it's a memory question
        workflow.add_edge("memory_retrieval", "check_memory_question")
        
        # Route based on question type: memory, page-specific, or regular
        def route_by_question_type(state: AgentState) -> str:
            if state.get("is_memory_question", False):
                return "memory_answer"
            elif state.get("is_page_question", False):
                return "page_retrieve"
            return "tool_detection"
        
        workflow.add_conditional_edges("check_memory_question", route_by_question_type)
        
        # Memory answer path goes directly to storage (skip retrieval/grading)
        workflow.add_edge("memory_answer", "memory_storage")
        
        # Page-specific path: retrieve page content -> generate answer -> store
        workflow.add_edge("page_retrieve", "page_answer")
        workflow.add_edge("page_answer", "memory_storage")
        
        # Regular path - skip tool detection, go straight to retrieval for PDF Q&A
        workflow.add_edge("tool_detection", "retrieve")
        workflow.add_edge("retrieve", "grade")
        
        def check_relevance(state: AgentState) -> str:
            if state.get("documents"):
                return "generate"  # Go directly to generate (skip tool execution for chat)
            if state.get("retries", 0) > 2:
                return "fail"
            return "rewrite"
        
        # Route: grade -> generate (if docs) or rewrite (if no docs)
        workflow.add_conditional_edges("grade", check_relevance)
        workflow.add_edge("rewrite", "retrieve")
        
        # Generate path (no tool execution in between)
        workflow.add_edge("generate", "reflect")
        workflow.add_edge("reflect", "web_search")
        workflow.add_edge("web_search", "memory_storage")
        workflow.add_edge("memory_storage", END)
        workflow.add_edge("fail", END)
        
        return workflow.compile()
    
    def invoke(
        self,
        question: str,
        domain: str,
        user_id: str,
        session_id: str
    ) -> Dict[str, Any]:
        """
        Invoke the agent with a question.
        
        Args:
            question: User question
            domain: Current domain (IT Service Desk, Developer Support, HR Operations)
            user_id: User identifier
            session_id: Session identifier
            
        Returns:
            Agent response with generation, tool_calls, and reasoning
        """
        try:
            result = self.app.invoke({
                "question": question,
                "original_question": question,
                "domain": domain,
                "user_id": user_id,
                "session_id": session_id,
                "documents": [],
                "generation": "",
                "is_grounded": False,
                "is_memory_question": False,
                "is_page_question": False,  # Added for page-specific queries
                "page_number": None,  # Added for page-specific queries
                "retries": 0,
                "memory_context": "",
                "long_term_memory": "",
                "should_store_memory": False,
                "tool_calls": [],
                "tool_result": {},  # Added for tool execution
                "reasoning_steps": [],
                "document_sources": [],
                "retrieval_results": [],
                "web_search_results": ""  # Added for web search
            })
            
            # Process documents for preview with source information
            processed_docs = []
            retrieval_results = result.get("retrieval_results", [])
            
            for i, doc in enumerate(result.get("documents", [])):
                doc_info = {
                    "content": doc,
                    "source": "Unknown",
                    "type": "text"
                }
                
                if i < len(retrieval_results):
                    doc_info["source"] = retrieval_results[i].get("source", "Unknown")
                    doc_info["type"] = retrieval_results[i].get("type", "text")
                    doc_info["metadata"] = retrieval_results[i].get("metadata", {})
                
                processed_docs.append(doc_info)
            
            return {
                "answer": result.get("generation", "No response generated."),
                "is_grounded": result.get("is_grounded", False),
                "tool_calls": result.get("tool_calls", []),
                "tool_result": result.get("tool_result", {}),  # Include tool result
                "reasoning_steps": result.get("reasoning_steps", []),
                "documents": processed_docs,
                "document_sources": result.get("document_sources", []),
                "web_search_results": result.get("web_search_results", "")  # Include web search
            }
            
        except Exception as e:
            return {
                "answer": f"❌ Agent error: {str(e)}",
                "is_grounded": False,
                "tool_calls": [],
                "tool_result": {},
                "reasoning_steps": [f"❌ Agent invocation failed: {str(e)}"],
                "documents": [],
                "document_sources": []
            }


# Singleton instance
_agent_instance = None

def get_agent(engine, memory_manager, groq_api_key: str = None) -> ByteMeAgent:
    """Get or create ByteMeAgent singleton"""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = ByteMeAgent(
            engine=engine,
            memory_manager=memory_manager,
            groq_api_key=groq_api_key
        )
    return _agent_instance
