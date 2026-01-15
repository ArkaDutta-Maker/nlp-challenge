"""
HR Operations Tool
Handles policy questions, leave applications, and HR processes
"""

import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional


class HROperationsTool:
    """
    HR Operations automation tool.
    Handles:
    - Policy questions and guidance
    - Leave application processing
    - Benefits information
    - Onboarding guidance
    """
    
    def __init__(self):
        # Leave balances (replace with HR system integration)
        self.leave_balances: Dict[str, Dict] = {}
        
        # Leave requests
        self.leave_requests: Dict[str, Dict] = {}
        
        # Policy documents
        self.policies = {
            "leave_policy": {
                "title": "Leave Policy",
                "effective_date": "2024-01-01",
                "content": {
                    "annual_leave": {
                        "entitlement": "20 days per year for full-time employees",
                        "accrual": "1.67 days per month",
                        "carryover": "Maximum 5 days can be carried to next year",
                        "notice": "Minimum 2 weeks notice for leaves > 5 days"
                    },
                    "sick_leave": {
                        "entitlement": "12 days per year",
                        "documentation": "Medical certificate required for > 2 consecutive days",
                        "notification": "Notify manager before shift starts"
                    },
                    "parental_leave": {
                        "maternity": "16 weeks paid leave",
                        "paternity": "4 weeks paid leave",
                        "eligibility": "After 1 year of continuous service"
                    }
                }
            },
            "remote_work": {
                "title": "Remote Work Policy",
                "effective_date": "2024-03-01",
                "content": {
                    "eligibility": "Employees with 6+ months tenure",
                    "frequency": "Up to 3 days per week",
                    "requirements": [
                        "Stable internet connection",
                        "Dedicated workspace",
                        "Available during core hours (10 AM - 4 PM)"
                    ],
                    "approval": "Manager approval required"
                }
            },
            "expense_policy": {
                "title": "Expense Reimbursement Policy",
                "effective_date": "2024-01-15",
                "content": {
                    "travel": {
                        "flights": "Economy class for domestic, business for > 6 hours international",
                        "hotels": "Up to $200/night domestic, $300/night international",
                        "meals": "$75/day domestic, $100/day international"
                    },
                    "equipment": {
                        "home_office": "Up to $500 one-time setup allowance",
                        "software": "Requires IT approval"
                    },
                    "submission": "Within 30 days of expense with receipts"
                }
            },
            "code_of_conduct": {
                "title": "Employee Code of Conduct",
                "effective_date": "2024-01-01",
                "content": {
                    "core_values": ["Integrity", "Respect", "Excellence", "Collaboration"],
                    "expectations": [
                        "Treat colleagues with respect and dignity",
                        "Maintain confidentiality of company information",
                        "Report conflicts of interest",
                        "Follow safety and security protocols"
                    ],
                    "reporting": "Report violations to HR or Ethics Hotline"
                }
            }
        }
        
        # Benefits information
        self.benefits = {
            "health_insurance": {
                "name": "Medical Insurance",
                "provider": "BlueCross BlueShield",
                "coverage": {
                    "employee": "100% premium covered",
                    "dependents": "80% premium covered",
                    "coverage_amount": "Up to $1M per year"
                },
                "enrollment": "Within 30 days of joining or during open enrollment (November)"
            },
            "dental": {
                "name": "Dental Insurance",
                "provider": "Delta Dental",
                "coverage": {
                    "preventive": "100% covered",
                    "basic": "80% covered",
                    "major": "50% covered",
                    "annual_max": "$2,000"
                }
            },
            "retirement": {
                "name": "401(k) Retirement Plan",
                "provider": "Fidelity",
                "details": {
                    "company_match": "100% match up to 6% of salary",
                    "vesting": "Immediate vesting for employee contributions, 3-year vesting for company match",
                    "enrollment": "Automatic at 3% after 90 days"
                }
            },
            "pto": {
                "name": "Paid Time Off",
                "details": {
                    "vacation": "20 days/year",
                    "sick": "12 days/year",
                    "personal": "3 days/year",
                    "holidays": "10 company holidays"
                }
            }
        }
    
    def get_policy(self, policy_name: str) -> Dict[str, Any]:
        """
        Get information about a specific policy.
        
        Args:
            policy_name: Name or keyword of the policy
        """
        policy_key = policy_name.lower().replace(" ", "_")
        
        for key, policy in self.policies.items():
            if key in policy_key or policy_key in key or policy_key in policy["title"].lower():
                return {
                    "success": True,
                    "policy": policy
                }
        
        return {
            "success": False,
            "message": f"Policy '{policy_name}' not found",
            "available_policies": [p["title"] for p in self.policies.values()]
        }
    
    def get_leave_balance(self, user_id: str) -> Dict[str, Any]:
        """
        Get leave balance for a user.
        
        Args:
            user_id: User identifier
        """
        # Initialize if not exists (demo purposes)
        if user_id not in self.leave_balances:
            self.leave_balances[user_id] = {
                "annual": 20.0,
                "sick": 12.0,
                "personal": 3.0,
                "parental": 0.0,
                "as_of": datetime.now().isoformat()
            }
        
        return {
            "success": True,
            "user_id": user_id,
            "balance": self.leave_balances[user_id]
        }
    
    def apply_leave(
        self,
        user_id: str,
        leave_type: str,
        start_date: str,
        end_date: str,
        reason: str = ""
    ) -> Dict[str, Any]:
        """
        Submit a leave application.
        
        Args:
            user_id: User identifier
            leave_type: Type of leave (annual, sick, personal, parental)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            reason: Reason for leave
        """
        request_id = f"LV{datetime.now().strftime('%Y%m%d')}{str(uuid.uuid4())[:6].upper()}"
        
        try:
            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d")
            days_requested = (end - start).days + 1
        except ValueError:
            return {
                "success": False,
                "message": "Invalid date format. Use YYYY-MM-DD"
            }
        
        # Check balance
        balance = self.get_leave_balance(user_id)["balance"]
        leave_type_key = leave_type.lower()
        
        if leave_type_key not in balance:
            return {
                "success": False,
                "message": f"Invalid leave type: {leave_type}",
                "valid_types": ["annual", "sick", "personal", "parental"]
            }
        
        if balance[leave_type_key] < days_requested:
            return {
                "success": False,
                "message": f"Insufficient {leave_type} leave balance",
                "available": balance[leave_type_key],
                "requested": days_requested
            }
        
        # Check notice period for longer leaves
        days_until_start = (start - datetime.now()).days
        if days_requested > 5 and days_until_start < 14:
            return {
                "success": False,
                "message": "Leaves > 5 days require minimum 2 weeks notice",
                "suggestion": "Please submit earlier or contact HR for exception approval"
            }
        
        # Create leave request
        request = {
            "id": request_id,
            "user_id": user_id,
            "leave_type": leave_type_key,
            "start_date": start_date,
            "end_date": end_date,
            "days": days_requested,
            "reason": reason,
            "status": "Pending Approval",
            "submitted_at": datetime.now().isoformat(),
            "approved_by": None
        }
        
        self.leave_requests[request_id] = request
        
        return {
            "success": True,
            "request": request,
            "message": f"Leave request {request_id} submitted successfully",
            "next_steps": [
                "Your manager will be notified for approval",
                "You'll receive an email once approved/rejected",
                "Update your calendar once approved"
            ]
        }
    
    def get_leave_requests(self, user_id: str) -> List[Dict]:
        """Get all leave requests for a user"""
        return [r for r in self.leave_requests.values() if r["user_id"] == user_id]
    
    def get_benefits_info(self, benefit_name: str = None) -> Dict[str, Any]:
        """
        Get benefits information.
        
        Args:
            benefit_name: Specific benefit to query (optional)
        """
        if benefit_name:
            benefit_key = benefit_name.lower().replace(" ", "_")
            
            for key, benefit in self.benefits.items():
                if key in benefit_key or benefit_key in key or benefit_key in benefit["name"].lower():
                    return {
                        "success": True,
                        "benefit": benefit
                    }
            
            return {
                "success": False,
                "message": f"Benefit '{benefit_name}' not found",
                "available_benefits": [b["name"] for b in self.benefits.values()]
            }
        
        # Return all benefits summary
        return {
            "success": True,
            "benefits_summary": {
                name: {"name": b["name"], "summary": list(b.get("coverage", b.get("details", {})).keys())}
                for name, b in self.benefits.items()
            }
        }
    
    def get_onboarding_checklist(self, user_id: str) -> Dict[str, Any]:
        """
        Get onboarding checklist for new employees.
        """
        return {
            "success": True,
            "user_id": user_id,
            "checklist": {
                "day_1": [
                    "✓ Complete I-9 and tax forms",
                    "✓ Receive employee ID badge",
                    "✓ Set up computer and email",
                    "✓ Review employee handbook",
                    "✓ Meet with HR for benefits overview"
                ],
                "week_1": [
                    "✓ Complete mandatory compliance training",
                    "✓ Enroll in benefits",
                    "✓ Set up direct deposit",
                    "✓ Meet with manager for role expectations",
                    "✓ Get access to required systems"
                ],
                "month_1": [
                    "✓ Complete all assigned training",
                    "✓ Attend new hire orientation session",
                    "✓ Schedule check-in with HR",
                    "✓ Complete 30-day manager review"
                ]
            },
            "contacts": {
                "hr_general": "hr@company.com",
                "benefits": "benefits@company.com",
                "it_support": "itsupport@company.com"
            }
        }
    
    def execute_action(self, action: str, parameters: Dict) -> Dict[str, Any]:
        """
        Execute a specific HR Operations action.
        
        Args:
            action: Action to perform
            parameters: Action parameters
        """
        actions = {
            "policy_query": lambda **p: self.get_policy(p.get("policy_name", "")),
            "leave_application": lambda **p: self.apply_leave(
                p.get("user_id", ""),
                p.get("leave_type", "annual"),
                p.get("start_date", ""),
                p.get("end_date", ""),
                p.get("reason", "")
            ),
            "leave_balance": lambda **p: self.get_leave_balance(p.get("user_id", "")),
            "benefits_info": lambda **p: self.get_benefits_info(p.get("benefit_name")),
            "onboarding": lambda **p: self.get_onboarding_checklist(p.get("user_id", ""))
        }
        
        if action in actions:
            try:
                return actions[action](**parameters)
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        return {"success": False, "error": f"Unknown action: {action}"}
