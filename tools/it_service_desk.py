"""
IT Service Desk Tool
Handles troubleshooting, ticket creation, and software requests
"""

import json
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List


class ITServiceDeskTool:
    """
    IT Service Desk automation tool.
    Handles:
    - Ticket creation and management
    - Troubleshooting workflows
    - Software requests
    - Password resets
    """
    
    def __init__(self):
        # In-memory ticket store (replace with database in production)
        self.tickets: Dict[str, Dict] = {}
        self.troubleshooting_guides = {
            "network": [
                "1. Check if your network cable is properly connected",
                "2. Restart your router/modem",
                "3. Run 'ipconfig /release' then 'ipconfig /renew' in CMD",
                "4. Check if other devices can connect to the network",
                "5. Contact IT if the issue persists"
            ],
            "password": [
                "1. Check if Caps Lock is off",
                "2. Try your previous password",
                "3. Use 'Forgot Password' on the login page",
                "4. Wait 15 minutes if account is locked",
                "5. Contact IT for manual reset if needed"
            ],
            "software": [
                "1. Restart the application",
                "2. Clear application cache/temp files",
                "3. Check for software updates",
                "4. Restart your computer",
                "5. Reinstall the application if issues persist"
            ],
            "printer": [
                "1. Check if printer is powered on and connected",
                "2. Clear any paper jams",
                "3. Restart the print spooler service",
                "4. Remove and re-add the printer",
                "5. Update printer drivers"
            ],
            "email": [
                "1. Check internet connectivity",
                "2. Verify email credentials",
                "3. Check sent/outbox for stuck emails",
                "4. Clear email cache",
                "5. Try webmail access to isolate the issue"
            ]
        }
        
        self.software_catalog = {
            "microsoft_office": {"name": "Microsoft Office 365", "approval_required": False, "install_time": "30 mins"},
            "vscode": {"name": "Visual Studio Code", "approval_required": False, "install_time": "10 mins"},
            "slack": {"name": "Slack", "approval_required": False, "install_time": "5 mins"},
            "zoom": {"name": "Zoom", "approval_required": False, "install_time": "5 mins"},
            "adobe_creative": {"name": "Adobe Creative Cloud", "approval_required": True, "install_time": "60 mins"},
            "vmware": {"name": "VMware Workstation", "approval_required": True, "install_time": "45 mins"},
            "docker": {"name": "Docker Desktop", "approval_required": True, "install_time": "20 mins"}
        }
    
    def create_ticket(
        self,
        user_id: str,
        issue: str,
        category: str = "general",
        priority: str = "medium",
        description: str = ""
    ) -> Dict[str, Any]:
        """
        Create a new IT support ticket.
        
        Args:
            user_id: User creating the ticket
            issue: Brief issue title
            category: network, software, hardware, access, other
            priority: low, medium, high, critical
            description: Detailed description
            
        Returns:
            Ticket details with ID
        """
        ticket_id = f"INC{datetime.now().strftime('%Y%m%d')}{str(uuid.uuid4())[:6].upper()}"
        
        ticket = {
            "id": ticket_id,
            "user_id": user_id,
            "issue": issue,
            "category": category,
            "priority": priority,
            "description": description,
            "status": "Open",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "assigned_to": None,
            "resolution": None,
            "sla": self._calculate_sla(priority)
        }
        
        self.tickets[ticket_id] = ticket
        
        return {
            "success": True,
            "ticket": ticket,
            "message": f"Ticket {ticket_id} created successfully. Expected resolution within {ticket['sla']}."
        }
    
    def get_ticket_status(self, ticket_id: str) -> Dict[str, Any]:
        """Get status of a specific ticket"""
        if ticket_id in self.tickets:
            return {
                "success": True,
                "ticket": self.tickets[ticket_id]
            }
        return {
            "success": False,
            "message": f"Ticket {ticket_id} not found"
        }
    
    def get_user_tickets(self, user_id: str) -> List[Dict]:
        """Get all tickets for a user"""
        return [t for t in self.tickets.values() if t["user_id"] == user_id]
    
    def get_troubleshooting_guide(self, category: str) -> Dict[str, Any]:
        """
        Get troubleshooting steps for a category.
        
        Args:
            category: Type of issue (network, password, software, printer, email)
        """
        # Find best matching category
        category_lower = category.lower()
        for key, steps in self.troubleshooting_guides.items():
            if key in category_lower or category_lower in key:
                return {
                    "success": True,
                    "category": key,
                    "steps": steps,
                    "note": "If these steps don't resolve your issue, please create a support ticket."
                }
        
        # Generic troubleshooting
        return {
            "success": True,
            "category": "general",
            "steps": [
                "1. Restart your computer",
                "2. Check for recent system updates",
                "3. Clear temporary files",
                "4. Check system resources (CPU, Memory, Disk)",
                "5. Create a support ticket if the issue persists"
            ],
            "note": "For specific troubleshooting, please provide more details about your issue."
        }
    
    def request_software(
        self,
        user_id: str,
        software_name: str,
        justification: str = ""
    ) -> Dict[str, Any]:
        """
        Request software installation.
        
        Args:
            user_id: Requesting user
            software_name: Name of software
            justification: Business justification
        """
        # Check catalog
        software_key = software_name.lower().replace(" ", "_")
        
        for key, info in self.software_catalog.items():
            if key in software_key or software_key in info["name"].lower():
                if info["approval_required"]:
                    # Create approval request ticket
                    ticket = self.create_ticket(
                        user_id=user_id,
                        issue=f"Software Request: {info['name']}",
                        category="software_request",
                        priority="low",
                        description=f"Justification: {justification}"
                    )
                    return {
                        "success": True,
                        "status": "pending_approval",
                        "software": info["name"],
                        "ticket_id": ticket["ticket"]["id"],
                        "message": f"Software request submitted. Requires manager approval. Ticket: {ticket['ticket']['id']}"
                    }
                else:
                    return {
                        "success": True,
                        "status": "approved",
                        "software": info["name"],
                        "install_time": info["install_time"],
                        "message": f"{info['name']} is pre-approved. Installation can proceed immediately. Estimated time: {info['install_time']}"
                    }
        
        # Unknown software
        ticket = self.create_ticket(
            user_id=user_id,
            issue=f"Software Request: {software_name}",
            category="software_request",
            priority="low",
            description=f"Justification: {justification}"
        )
        return {
            "success": True,
            "status": "review_required",
            "software": software_name,
            "ticket_id": ticket["ticket"]["id"],
            "message": f"Software not in catalog. Request submitted for review. Ticket: {ticket['ticket']['id']}"
        }
    
    def initiate_password_reset(self, user_id: str, target_system: str = "AD") -> Dict[str, Any]:
        """
        Initiate password reset process.
        
        Args:
            user_id: User requesting reset
            target_system: System for reset (AD, Email, VPN, etc.)
        """
        reset_token = str(uuid.uuid4())[:8].upper()
        
        return {
            "success": True,
            "reset_token": reset_token,
            "target_system": target_system,
            "instructions": [
                f"1. A password reset link has been sent to your registered email",
                f"2. Reset token: {reset_token} (valid for 30 minutes)",
                "3. Choose a password with at least 12 characters, including uppercase, lowercase, number, and special character",
                "4. Your new password cannot match your last 5 passwords",
                "5. Contact IT if you don't receive the email within 5 minutes"
            ],
            "message": f"Password reset initiated for {target_system}. Check your email for reset instructions."
        }
    
    def _calculate_sla(self, priority: str) -> str:
        """Calculate SLA based on priority"""
        sla_map = {
            "critical": "4 hours",
            "high": "8 hours",
            "medium": "24 hours",
            "low": "72 hours"
        }
        return sla_map.get(priority, "24 hours")
    
    def execute_action(self, action: str, parameters: Dict) -> Dict[str, Any]:
        """
        Execute a specific IT Service Desk action.
        
        Args:
            action: Action to perform
            parameters: Action parameters
        """
        actions = {
            "create_ticket": self.create_ticket,
            "check_status": lambda **p: self.get_ticket_status(p.get("ticket_id", "")),
            "password_reset": self.initiate_password_reset,
            "software_request": self.request_software,
            "troubleshoot": lambda **p: self.get_troubleshooting_guide(p.get("category", "general"))
        }
        
        if action in actions:
            try:
                return actions[action](**parameters)
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        return {"success": False, "error": f"Unknown action: {action}"}
