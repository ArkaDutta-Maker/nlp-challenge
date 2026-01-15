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
    generation: str
    is_grounded: bool
    retries: int
    memory_context: str
    long_term_memory: str
    should_store_memory: bool
    tool_calls: List[Dict]  # For tracking tool usage
    reasoning_steps: List[str]  # For inspector panel


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
            "IT Service Desk": """You are an IT Service Desk assistant. You help with:
- Troubleshooting technical issues
- Creating support tickets
- Software installation requests
- Password resets and access issues
- Network and connectivity problems
Be professional, follow ITIL best practices, and always offer to create a ticket if the issue cannot be resolved immediately.""",
            
            "Developer Support": """You are a Developer Support assistant. You help with:
- Explaining legacy code and documentation
- Suggesting code fixes and improvements
- Debugging assistance
- API documentation and usage
- Best practices and code review
Provide code examples when helpful and explain technical concepts clearly.""",
            
            "HR Operations": """You are an HR Operations assistant. You help with:
- Company policy questions
- Leave application guidance
- Benefits information
- Onboarding procedures
- Performance review processes
Be empathetic, maintain confidentiality, and direct sensitive matters to HR personnel when appropriate."""
        }
        
        # RAG Chain with Memory Context
        self.rag_prompt = PromptTemplate(
            template="""You are a helpful enterprise assistant with access to document context and conversation history.

{domain_system_prompt}

📚 DOCUMENT CONTEXT:
{context}

🧠 RECENT CONVERSATION:
{memory_context}

🧠 RELEVANT PAST CONVERSATIONS:
{long_term_memory}

❓ CURRENT QUESTION: {question}

Instructions:
- Use the document context as your primary source
- Reference conversation history to maintain context continuity
- If the question refers to previous answers, use the memory
- Be concise, accurate, and professional
- If you cannot find relevant information, say so honestly

Answer:""",
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
        
        # Tool Detection Chain
        self.tool_prompt = PromptTemplate(
            template="""Analyze the user's question and determine if a specific action/tool is needed.

Domain: {domain}
Question: {question}

For IT Service Desk, detect:
- "create_ticket": User wants to create a support ticket
- "check_status": User wants to check ticket status
- "password_reset": User needs password reset
- "software_request": User needs software installation

For Developer Support, detect:
- "code_explanation": User wants code explained
- "suggest_fix": User wants bug fix suggestions
- "api_docs": User needs API documentation

For HR Operations, detect:
- "leave_application": User wants to apply for leave
- "policy_query": User asking about policies
- "benefits_info": User asking about benefits

Return JSON with "tool" and "parameters" fields, or {{"tool": "none"}} if no specific action needed.
Example: {{"tool": "create_ticket", "parameters": {{"issue": "...", "priority": "medium"}}}}

Response:""",
            input_variables=["domain", "question"]
        )
        self.tool_chain = self.tool_prompt | self.llm_router | StrOutputParser()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        
        # Define nodes
        def memory_retrieval_node(state: AgentState) -> Dict:
            """Retrieve relevant memories"""
            reasoning = ["🧠 Retrieving relevant memories..."]
            
            context = self.memory_manager.get_context(
                session_id=state["session_id"],
                user_id=state["user_id"],
                query=state["question"],
                domain=state["domain"]
            )
            
            reasoning.append(f"   - Short-term: {len(context['short_term'])} chars")
            reasoning.append(f"   - Long-term: {len(context['long_term'])} chars")
            
            return {
                "memory_context": context["short_term"],
                "long_term_memory": context["long_term"],
                "original_question": state.get("original_question") or state["question"],
                "reasoning_steps": state.get("reasoning_steps", []) + reasoning
            }
        
        def tool_detection_node(state: AgentState) -> Dict:
            """Detect if a specific tool/action is needed"""
            reasoning = ["🔧 Detecting required tools..."]
            
            try:
                raw_res = self.tool_chain.invoke({
                    "domain": state["domain"],
                    "question": state["question"]
                })
                tool_info = parse_json_safe(raw_res)
                
                if tool_info.get("tool") and tool_info["tool"] != "none":
                    reasoning.append(f"   - Detected tool: {tool_info['tool']}")
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
            
            results = self.engine.hybrid_search(
                query=state["question"],
                domain=state["domain"],
                k=5
            )
            
            docs = [r["content"] for r in results]
            reasoning.append(f"   - Found {len(docs)} documents")
            
            return {
                "documents": docs,
                "retries": state.get("retries", 0),
                "reasoning_steps": state.get("reasoning_steps", []) + reasoning
            }
        
        def grade_node(state: AgentState) -> Dict:
            """Grade retrieved documents for relevance"""
            reasoning = ["📝 Grading document relevance..."]
            relevant = []
            
            if not state["documents"]:
                reasoning.append("   - No documents to grade")
                return {
                    "documents": [],
                    "retries": state["retries"] + 1,
                    "reasoning_steps": state.get("reasoning_steps", []) + reasoning
                }
            
            for i, doc in enumerate(state["documents"]):
                try:
                    raw_res = self.grader_chain.invoke({
                        "question": state["question"],
                        "document": doc[:500]
                    })
                    parsed_res = parse_json_safe(raw_res)
                    
                    if parsed_res.get("score") == "yes":
                        relevant.append(doc)
                        reasoning.append(f"   - Doc {i+1}: ✓ Relevant")
                    else:
                        reasoning.append(f"   - Doc {i+1}: ✗ Not relevant")
                except:
                    continue
            
            new_retries = state["retries"] + 1 if not relevant else state["retries"]
            reasoning.append(f"   - {len(relevant)}/{len(state['documents'])} documents passed")
            
            return {
                "documents": relevant,
                "retries": new_retries,
                "reasoning_steps": state.get("reasoning_steps", []) + reasoning
            }
        
        def rewrite_node(state: AgentState) -> Dict:
            """Rewrite query for better retrieval"""
            reasoning = ["🔄 Rewriting query for better results..."]
            
            new_q = self.rewriter_chain.invoke({"question": state["question"]})
            new_q = new_q.strip().replace('"', '')
            
            reasoning.append(f"   - New query: '{new_q[:50]}...'")
            
            return {
                "question": new_q,
                "reasoning_steps": state.get("reasoning_steps", []) + reasoning
            }
        
        def generate_node(state: AgentState) -> Dict:
            """Generate answer using RAG"""
            reasoning = ["💡 Generating answer..."]
            
            domain_prompt = self.domain_prompts.get(
                state["domain"],
                "You are a helpful enterprise assistant."
            )
            
            gen = self.rag_chain.invoke({
                "domain_system_prompt": domain_prompt,
                "context": "\n\n".join(state["documents"]),
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
        
        def graceful_fail_node(state: AgentState) -> Dict:
            """Handle max retries"""
            reasoning = ["❌ Max retries reached"]
            
            return {
                "generation": "I apologize, but I couldn't find relevant information to answer your question. Please try rephrasing or provide more details.",
                "should_store_memory": False,
                "is_grounded": False,
                "reasoning_steps": state.get("reasoning_steps", []) + reasoning
            }
        
        def reflection_node(state: AgentState) -> Dict:
            """Verify answer grounding"""
            reasoning = ["🛡️ Verifying answer grounding..."]
            
            if "I apologize" in state["generation"] or "couldn't find" in state["generation"]:
                reasoning.append("   - Fallback response detected")
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
            
            reasoning.append("   - ⚠ Answer may not be fully grounded")
            return {
                "is_grounded": False,
                "generation": state["generation"] + "\n\n⚠️ Note: This response may not be fully verified against the source documents.",
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
        workflow.add_node("tool_detection", tool_detection_node)
        workflow.add_node("retrieve", retrieve_node)
        workflow.add_node("grade", grade_node)
        workflow.add_node("rewrite", rewrite_node)
        workflow.add_node("generate", generate_node)
        workflow.add_node("reflect", reflection_node)
        workflow.add_node("memory_storage", memory_storage_node)
        workflow.add_node("fail", graceful_fail_node)
        
        # Set entry point
        workflow.set_entry_point("memory_retrieval")
        
        # Add edges
        workflow.add_edge("memory_retrieval", "tool_detection")
        workflow.add_edge("tool_detection", "retrieve")
        workflow.add_edge("retrieve", "grade")
        
        def check_relevance(state: AgentState) -> str:
            if state["documents"]:
                return "generate"
            if state["retries"] > 2:
                return "fail"
            return "rewrite"
        
        workflow.add_conditional_edges("grade", check_relevance)
        workflow.add_edge("rewrite", "retrieve")
        workflow.add_edge("generate", "reflect")
        workflow.add_edge("reflect", "memory_storage")
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
        result = self.app.invoke({
            "question": question,
            "original_question": question,
            "domain": domain,
            "user_id": user_id,
            "session_id": session_id,
            "documents": [],
            "generation": "",
            "is_grounded": False,
            "retries": 0,
            "memory_context": "",
            "long_term_memory": "",
            "should_store_memory": False,
            "tool_calls": [],
            "reasoning_steps": []
        })
        
        return {
            "answer": result.get("generation", "No response generated."),
            "is_grounded": result.get("is_grounded", False),
            "tool_calls": result.get("tool_calls", []),
            "reasoning_steps": result.get("reasoning_steps", []),
            "documents": result.get("documents", [])
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
