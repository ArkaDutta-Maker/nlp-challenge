"""
Developer Support Tool
Handles code documentation, fixes, and API support
"""

import json
from datetime import datetime
from typing import Dict, Any, List, Optional


class DeveloperSupportTool:
    """
    Developer Support automation tool.
    Handles:
    - Legacy code documentation retrieval
    - Code fix suggestions
    - API documentation
    - Best practices guidance
    """
    
    def __init__(self):
        # Sample legacy code documentation (replace with actual database)
        self.code_docs = {
            "auth_module": {
                "name": "Authentication Module",
                "version": "2.3.1",
                "language": "Python",
                "description": "Handles user authentication using JWT tokens",
                "functions": {
                    "authenticate_user": {
                        "signature": "authenticate_user(username: str, password: str) -> dict",
                        "description": "Validates user credentials and returns JWT token",
                        "parameters": ["username: User's login name", "password: User's password"],
                        "returns": "Dict with 'token' and 'expires_at' fields",
                        "example": "result = authenticate_user('john', 'pass123')"
                    },
                    "verify_token": {
                        "signature": "verify_token(token: str) -> bool",
                        "description": "Validates JWT token and checks expiration",
                        "parameters": ["token: JWT token string"],
                        "returns": "Boolean indicating token validity"
                    }
                },
                "dependencies": ["PyJWT", "bcrypt", "redis"],
                "last_updated": "2024-06-15"
            },
            "data_pipeline": {
                "name": "Data Pipeline Module",
                "version": "1.8.0",
                "language": "Python",
                "description": "ETL pipeline for processing customer data",
                "functions": {
                    "extract_data": {
                        "signature": "extract_data(source: str, query: str) -> DataFrame",
                        "description": "Extracts data from specified source",
                        "parameters": ["source: Database connection name", "query: SQL query"],
                        "returns": "Pandas DataFrame with query results"
                    },
                    "transform_data": {
                        "signature": "transform_data(df: DataFrame, rules: dict) -> DataFrame",
                        "description": "Applies transformation rules to dataframe",
                        "parameters": ["df: Input DataFrame", "rules: Transformation rules dict"],
                        "returns": "Transformed DataFrame"
                    }
                },
                "dependencies": ["pandas", "sqlalchemy", "apache-airflow"],
                "last_updated": "2024-08-20"
            }
        }
        
        # Common code fixes
        self.common_fixes = {
            "null_pointer": {
                "description": "Null Pointer / None Reference Error",
                "languages": ["Python", "Java", "JavaScript"],
                "solution": "Add null checks before accessing object properties",
                "python_example": """
# Before (problematic)
result = obj.property.value

# After (fixed)
if obj and obj.property:
    result = obj.property.value
else:
    result = default_value
""",
                "prevention": "Use Optional type hints and implement defensive programming"
            },
            "memory_leak": {
                "description": "Memory Leak Issues",
                "languages": ["Python", "Java", "JavaScript"],
                "solution": "Properly close resources and use context managers",
                "python_example": """
# Before (problematic)
f = open('file.txt', 'r')
data = f.read()
# File never closed!

# After (fixed)
with open('file.txt', 'r') as f:
    data = f.read()
# File automatically closed
""",
                "prevention": "Always use context managers (with statements) for resources"
            },
            "sql_injection": {
                "description": "SQL Injection Vulnerability",
                "languages": ["Python", "Java", "PHP"],
                "solution": "Use parameterized queries instead of string concatenation",
                "python_example": """
# Before (vulnerable)
query = f"SELECT * FROM users WHERE id = {user_id}"

# After (secure)
query = "SELECT * FROM users WHERE id = %s"
cursor.execute(query, (user_id,))
""",
                "prevention": "Never concatenate user input into SQL queries"
            },
            "race_condition": {
                "description": "Race Condition in Concurrent Code",
                "languages": ["Python", "Java", "Go"],
                "solution": "Use proper synchronization mechanisms",
                "python_example": """
# Before (race condition)
counter = 0
def increment():
    global counter
    counter += 1

# After (thread-safe)
import threading
counter = 0
lock = threading.Lock()

def increment():
    global counter
    with lock:
        counter += 1
""",
                "prevention": "Use locks, semaphores, or thread-safe data structures"
            }
        }
        
        # API documentation
        self.api_docs = {
            "user_api": {
                "name": "User Management API",
                "base_url": "/api/v1/users",
                "endpoints": [
                    {
                        "method": "GET",
                        "path": "/",
                        "description": "List all users",
                        "parameters": ["page: int (optional)", "limit: int (optional)"],
                        "response": "{ 'users': [...], 'total': int }"
                    },
                    {
                        "method": "GET",
                        "path": "/{id}",
                        "description": "Get user by ID",
                        "parameters": ["id: int (required)"],
                        "response": "{ 'id': int, 'name': str, 'email': str }"
                    },
                    {
                        "method": "POST",
                        "path": "/",
                        "description": "Create new user",
                        "body": "{ 'name': str, 'email': str, 'role': str }",
                        "response": "{ 'id': int, 'created_at': datetime }"
                    }
                ],
                "authentication": "Bearer token required in Authorization header"
            }
        }
    
    def get_code_documentation(self, module_name: str) -> Dict[str, Any]:
        """
        Get documentation for a legacy code module.
        
        Args:
            module_name: Name or keyword of the module
        """
        module_key = module_name.lower().replace(" ", "_")
        
        for key, doc in self.code_docs.items():
            if key in module_key or module_key in key or module_key in doc["name"].lower():
                return {
                    "success": True,
                    "documentation": doc
                }
        
        return {
            "success": False,
            "message": f"Documentation for '{module_name}' not found",
            "available_modules": list(self.code_docs.keys())
        }
    
    def suggest_fix(self, issue_type: str, code_snippet: str = "") -> Dict[str, Any]:
        """
        Suggest fix for a common coding issue.
        
        Args:
            issue_type: Type of issue (null_pointer, memory_leak, sql_injection, race_condition)
            code_snippet: Optional problematic code snippet
        """
        issue_key = issue_type.lower().replace(" ", "_").replace("-", "_")
        
        for key, fix in self.common_fixes.items():
            if key in issue_key or issue_key in key:
                return {
                    "success": True,
                    "issue": fix["description"],
                    "solution": fix["solution"],
                    "example": fix["python_example"],
                    "prevention": fix["prevention"]
                }
        
        return {
            "success": False,
            "message": f"No specific fix found for '{issue_type}'",
            "available_fixes": list(self.common_fixes.keys()),
            "suggestion": "Please describe your issue in more detail or create a support ticket."
        }
    
    def get_api_documentation(self, api_name: str) -> Dict[str, Any]:
        """
        Get API documentation.
        
        Args:
            api_name: Name or keyword of the API
        """
        api_key = api_name.lower().replace(" ", "_")
        
        for key, doc in self.api_docs.items():
            if key in api_key or api_key in key or api_key in doc["name"].lower():
                return {
                    "success": True,
                    "api_documentation": doc
                }
        
        return {
            "success": False,
            "message": f"API documentation for '{api_name}' not found",
            "available_apis": list(self.api_docs.keys())
        }
    
    def code_review_checklist(self, language: str = "python") -> Dict[str, Any]:
        """
        Get code review checklist for a language.
        """
        checklists = {
            "python": [
                "✓ Follow PEP 8 style guidelines",
                "✓ Use type hints for function signatures",
                "✓ Write docstrings for public functions/classes",
                "✓ Handle exceptions appropriately",
                "✓ Use context managers for resources",
                "✓ Avoid mutable default arguments",
                "✓ Use list comprehensions where appropriate",
                "✓ Write unit tests for new code",
                "✓ Check for security vulnerabilities",
                "✓ Review for performance optimizations"
            ],
            "javascript": [
                "✓ Use const/let instead of var",
                "✓ Handle promises/async properly",
                "✓ Avoid callback hell",
                "✓ Use strict equality (===)",
                "✓ Sanitize user inputs",
                "✓ Handle errors in async code",
                "✓ Use modern ES6+ features",
                "✓ Write unit tests",
                "✓ Check for XSS vulnerabilities",
                "✓ Review bundle size impact"
            ],
            "java": [
                "✓ Follow Java naming conventions",
                "✓ Use appropriate access modifiers",
                "✓ Handle exceptions properly",
                "✓ Close resources in finally/try-with-resources",
                "✓ Avoid raw types in generics",
                "✓ Use interfaces for abstraction",
                "✓ Write JavaDoc comments",
                "✓ Write unit tests",
                "✓ Check for thread safety",
                "✓ Review for memory leaks"
            ]
        }
        
        lang = language.lower()
        if lang in checklists:
            return {
                "success": True,
                "language": lang,
                "checklist": checklists[lang]
            }
        
        return {
            "success": True,
            "language": "general",
            "checklist": [
                "✓ Code is readable and well-documented",
                "✓ Functions are single-purpose and small",
                "✓ Error handling is comprehensive",
                "✓ No security vulnerabilities",
                "✓ Unit tests are included",
                "✓ No hard-coded credentials",
                "✓ Logging is appropriate",
                "✓ Performance is acceptable"
            ]
        }
    
    def execute_action(self, action: str, parameters: Dict) -> Dict[str, Any]:
        """
        Execute a specific Developer Support action.
        
        Args:
            action: Action to perform
            parameters: Action parameters
        """
        actions = {
            "code_explanation": lambda **p: self.get_code_documentation(p.get("module", "")),
            "suggest_fix": lambda **p: self.suggest_fix(p.get("issue_type", ""), p.get("code", "")),
            "api_docs": lambda **p: self.get_api_documentation(p.get("api_name", "")),
            "code_review": lambda **p: self.code_review_checklist(p.get("language", "python"))
        }
        
        if action in actions:
            try:
                return actions[action](**parameters)
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        return {"success": False, "error": f"Unknown action: {action}"}
