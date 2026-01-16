"""
IT Service Desk Tool - Enhanced with RAG Context Support
Handles troubleshooting, ticket creation, software requests, and knowledge-based support
"""

import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List


class ITServiceDeskTool:
    """
    IT Service Desk automation tool with RAG context integration.
    
    Capabilities:
    - Intelligent ticket creation with auto-categorization
    - Context-aware troubleshooting using knowledge base
    - Software request management with approval workflows
    - Password reset and access management
    - System status checking
    - Escalation handling
    """
    
    def __init__(self):
        # In-memory ticket store (replace with database in production)
        self.tickets: Dict[str, Dict] = {}
        self.active_incidents: Dict[str, Dict] = {}
        
        # Enhanced troubleshooting knowledge base
        self.troubleshooting_guides = {
            "network": {
                "title": "Network Connectivity Issues",
                "symptoms": ["cannot connect to internet", "slow network", "wifi issues", "VPN not connecting"],
                "steps": [
                    {"step": 1, "action": "Check physical connections", "detail": "Ensure network cable is properly connected or WiFi is enabled"},
                    {"step": 2, "action": "Restart network adapter", "detail": "Disable and re-enable network adapter in Network Connections"},
                    {"step": 3, "action": "Release/Renew IP", "detail": "Run 'ipconfig /release' then 'ipconfig /renew' in Command Prompt"},
                    {"step": 4, "action": "Flush DNS cache", "detail": "Run 'ipconfig /flushdns' in Command Prompt"},
                    {"step": 5, "action": "Check proxy settings", "detail": "Ensure proxy settings are correct in Internet Options"},
                    {"step": 6, "action": "Test with other devices", "detail": "Verify if other devices can connect to the network"},
                    {"step": 7, "action": "Contact IT Support", "detail": "If issue persists, create a support ticket"}
                ],
                "common_solutions": {
                    "vpn": "Restart VPN client, check credentials, try alternate VPN server",
                    "wifi": "Forget and reconnect to WiFi network, restart router if possible",
                    "ethernet": "Try different port, check cable integrity, test with different cable"
                }
            },
            "password": {
                "title": "Password & Authentication Issues",
                "symptoms": ["forgot password", "account locked", "cannot login", "password expired"],
                "steps": [
                    {"step": 1, "action": "Verify Caps Lock", "detail": "Ensure Caps Lock is not accidentally enabled"},
                    {"step": 2, "action": "Try previous password", "detail": "If recently changed, try your previous password"},
                    {"step": 3, "action": "Wait for lockout", "detail": "If locked out, wait 15-30 minutes for auto-unlock"},
                    {"step": 4, "action": "Use self-service portal", "detail": "Go to password.company.com for self-service reset"},
                    {"step": 5, "action": "Check email for reset link", "detail": "Request password reset email if available"},
                    {"step": 6, "action": "Contact IT Helpdesk", "detail": "Call IT Helpdesk for manual reset if needed"}
                ],
                "password_policy": {
                    "min_length": 12,
                    "requirements": ["uppercase", "lowercase", "number", "special character"],
                    "history": "Cannot reuse last 10 passwords",
                    "expiry": "90 days"
                }
            },
            "software": {
                "title": "Software & Application Issues",
                "symptoms": ["application crash", "software not opening", "error message", "slow application"],
                "steps": [
                    {"step": 1, "action": "Restart application", "detail": "Close and reopen the application"},
                    {"step": 2, "action": "Check for updates", "detail": "Ensure you have the latest version installed"},
                    {"step": 3, "action": "Clear cache", "detail": "Clear application cache and temporary files"},
                    {"step": 4, "action": "Restart computer", "detail": "Perform a full system restart"},
                    {"step": 5, "action": "Repair installation", "detail": "Use the repair option in Programs and Features"},
                    {"step": 6, "action": "Reinstall", "detail": "Uninstall and reinstall the application"},
                    {"step": 7, "action": "Check compatibility", "detail": "Verify system requirements are met"}
                ],
                "common_fixes": {
                    "office": "Run Office repair from Control Panel > Programs",
                    "browser": "Clear browser cache, disable extensions, reset browser settings",
                    "outlook": "Run scanpst.exe to repair PST file, recreate Outlook profile"
                }
            },
            "printer": {
                "title": "Printer Issues",
                "symptoms": ["cannot print", "printer offline", "print queue stuck", "poor print quality"],
                "steps": [
                    {"step": 1, "action": "Check printer status", "detail": "Ensure printer is powered on and shows ready status"},
                    {"step": 2, "action": "Check connections", "detail": "Verify USB/network connection is secure"},
                    {"step": 3, "action": "Clear print queue", "detail": "Cancel all pending print jobs"},
                    {"step": 4, "action": "Restart Print Spooler", "detail": "Run 'net stop spooler && net start spooler' as admin"},
                    {"step": 5, "action": "Reinstall printer", "detail": "Remove printer and add it again"},
                    {"step": 6, "action": "Update drivers", "detail": "Download and install latest drivers from manufacturer"}
                ]
            },
            "email": {
                "title": "Email Issues",
                "symptoms": ["cannot send email", "not receiving email", "outlook not syncing", "attachment issues"],
                "steps": [
                    {"step": 1, "action": "Check internet", "detail": "Verify internet connectivity"},
                    {"step": 2, "action": "Try webmail", "detail": "Access email via web browser to isolate the issue"},
                    {"step": 3, "action": "Check outbox", "detail": "Look for stuck emails in outbox/drafts"},
                    {"step": 4, "action": "Repair profile", "detail": "Repair or recreate email profile"},
                    {"step": 5, "action": "Check storage", "detail": "Verify mailbox is not full"},
                    {"step": 6, "action": "Disable add-ins", "detail": "Start Outlook in safe mode to test"}
                ]
            },
            "hardware": {
                "title": "Hardware Issues",
                "symptoms": ["computer slow", "blue screen", "strange noises", "overheating", "screen issues"],
                "steps": [
                    {"step": 1, "action": "Restart computer", "detail": "Perform a clean restart"},
                    {"step": 2, "action": "Check temperature", "detail": "Ensure adequate ventilation, clean dust from vents"},
                    {"step": 3, "action": "Run diagnostics", "detail": "Use built-in hardware diagnostics"},
                    {"step": 4, "action": "Check disk health", "detail": "Run chkdsk and check SMART status"},
                    {"step": 5, "action": "Update drivers", "detail": "Ensure all device drivers are current"},
                    {"step": 6, "action": "Schedule replacement", "detail": "If hardware failure suspected, request replacement"}
                ]
            },
            "vpn": {
                "title": "VPN Connection Issues",
                "symptoms": ["vpn not connecting", "vpn disconnects", "slow vpn", "cannot access resources on vpn"],
                "steps": [
                    {"step": 1, "action": "Check internet", "detail": "Ensure base internet connection is working"},
                    {"step": 2, "action": "Restart VPN client", "detail": "Close and reopen VPN application"},
                    {"step": 3, "action": "Try different server", "detail": "Select alternate VPN server location"},
                    {"step": 4, "action": "Check credentials", "detail": "Verify VPN username and password"},
                    {"step": 5, "action": "Disable firewall temporarily", "detail": "Test if firewall is blocking VPN"},
                    {"step": 6, "action": "Reinstall VPN client", "detail": "Download fresh copy and reinstall"}
                ]
            }
        }
        
        # Enhanced software catalog with categories
        self.software_catalog = {
            "productivity": {
                "microsoft_office": {"name": "Microsoft Office 365", "approval_required": False, "install_time": "30 mins", "license": "enterprise"},
                "slack": {"name": "Slack", "approval_required": False, "install_time": "5 mins", "license": "enterprise"},
                "zoom": {"name": "Zoom", "approval_required": False, "install_time": "5 mins", "license": "enterprise"},
                "teams": {"name": "Microsoft Teams", "approval_required": False, "install_time": "10 mins", "license": "enterprise"},
            },
            "development": {
                "vscode": {"name": "Visual Studio Code", "approval_required": False, "install_time": "10 mins", "license": "free"},
                "git": {"name": "Git", "approval_required": False, "install_time": "5 mins", "license": "free"},
                "docker": {"name": "Docker Desktop", "approval_required": True, "install_time": "20 mins", "license": "enterprise"},
                "postman": {"name": "Postman", "approval_required": False, "install_time": "10 mins", "license": "free"},
                "python": {"name": "Python", "approval_required": False, "install_time": "10 mins", "license": "free"},
                "nodejs": {"name": "Node.js", "approval_required": False, "install_time": "10 mins", "license": "free"},
            },
            "security": {
                "antivirus": {"name": "Enterprise Antivirus", "approval_required": False, "install_time": "15 mins", "license": "enterprise"},
                "vpn_client": {"name": "Corporate VPN Client", "approval_required": False, "install_time": "10 mins", "license": "enterprise"},
            },
            "specialized": {
                "adobe_creative": {"name": "Adobe Creative Cloud", "approval_required": True, "install_time": "60 mins", "license": "per-seat"},
                "vmware": {"name": "VMware Workstation", "approval_required": True, "install_time": "45 mins", "license": "per-seat"},
                "tableau": {"name": "Tableau Desktop", "approval_required": True, "install_time": "30 mins", "license": "per-seat"},
            }
        }
        
        # System status tracker
        self.system_status = {
            "email_server": {"status": "operational", "last_check": datetime.now()},
            "vpn": {"status": "operational", "last_check": datetime.now()},
            "active_directory": {"status": "operational", "last_check": datetime.now()},
            "file_server": {"status": "operational", "last_check": datetime.now()},
            "intranet": {"status": "operational", "last_check": datetime.now()},
        }
        
        # Priority and SLA definitions
        self.priority_config = {
            "critical": {"sla_hours": 4, "description": "System down, affecting multiple users"},
            "high": {"sla_hours": 8, "description": "Major functionality impaired"},
            "medium": {"sla_hours": 24, "description": "Standard request or issue"},
            "low": {"sla_hours": 72, "description": "Minor issue or enhancement request"}
        }
    
    def create_ticket(
        self,
        user_id: str,
        issue: str,
        category: str = "general",
        priority: str = "medium",
        description: str = "",
        context: str = ""
    ) -> Dict[str, Any]:
        """
        Create a new IT support ticket with intelligent categorization.
        
        Args:
            user_id: User creating the ticket
            issue: Brief issue title
            category: network, software, hardware, access, email, other
            priority: low, medium, high, critical
            description: Detailed description
            context: Additional context from RAG retrieval
            
        Returns:
            Ticket details with ID and recommended actions
        """
        ticket_id = f"INC{datetime.now().strftime('%Y%m%d')}{str(uuid.uuid4())[:6].upper()}"
        
        # Auto-categorize if not specified
        if category == "general":
            category = self._auto_categorize(issue + " " + description)
        
        # Determine priority if context suggests urgency
        if any(word in (issue + description).lower() for word in ["urgent", "critical", "down", "outage", "cannot work"]):
            priority = "high" if priority == "medium" else priority
        
        sla_info = self.priority_config.get(priority, self.priority_config["medium"])
        sla_deadline = datetime.now() + timedelta(hours=sla_info["sla_hours"])
        
        ticket = {
            "id": ticket_id,
            "user_id": user_id,
            "issue": issue,
            "category": category,
            "priority": priority,
            "description": description,
            "context": context,
            "status": "Open",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "assigned_to": self._auto_assign(category, priority),
            "resolution": None,
            "sla_deadline": sla_deadline.isoformat(),
            "sla_hours": sla_info["sla_hours"],
            "escalation_level": 0,
            "notes": []
        }
        
        self.tickets[ticket_id] = ticket
        
        # Get relevant troubleshooting steps if available
        initial_steps = self._get_initial_recommendations(category, issue)
        
        return {
            "success": True,
            "ticket": ticket,
            "message": f"✅ Ticket {ticket_id} created successfully",
            "sla": f"Expected resolution within {sla_info['sla_hours']} hours",
            "assigned_to": ticket["assigned_to"],
            "initial_recommendations": initial_steps,
            "next_steps": "An IT support specialist will review your ticket shortly. You can track status using your ticket ID."
        }
    
    def get_ticket_status(self, ticket_id: str = None, user_id: str = None) -> Dict[str, Any]:
        """Get status of a specific ticket or all user tickets"""
        if ticket_id:
            if ticket_id in self.tickets:
                ticket = self.tickets[ticket_id]
                # Calculate SLA status
                sla_deadline = datetime.fromisoformat(ticket["sla_deadline"])
                time_remaining = sla_deadline - datetime.now()
                sla_status = "On Track" if time_remaining.total_seconds() > 0 else "SLA Breached"
                
                return {
                    "success": True,
                    "ticket": ticket,
                    "sla_status": sla_status,
                    "time_remaining": str(time_remaining) if time_remaining.total_seconds() > 0 else "Overdue"
                }
            return {
                "success": False,
                "message": f"Ticket {ticket_id} not found. Please verify the ticket ID."
            }
        elif user_id:
            user_tickets = [t for t in self.tickets.values() if t["user_id"] == user_id]
            return {
                "success": True,
                "tickets": user_tickets,
                "total": len(user_tickets),
                "open": len([t for t in user_tickets if t["status"] == "Open"]),
                "in_progress": len([t for t in user_tickets if t["status"] == "In Progress"]),
                "resolved": len([t for t in user_tickets if t["status"] == "Resolved"])
            }
        return {"success": False, "message": "Please provide ticket_id or user_id"}
    
    def troubleshoot(
        self,
        category: str,
        symptoms: str = "",
        context: str = ""
    ) -> Dict[str, Any]:
        """
        Get intelligent troubleshooting guidance with RAG context.
        
        Args:
            category: Type of issue
            symptoms: Specific symptoms described
            context: Additional context from knowledge base
        """
        # Find best matching category
        category_lower = category.lower()
        matched_guide = None
        
        for key, guide in self.troubleshooting_guides.items():
            if key in category_lower:
                matched_guide = guide
                break
            # Also check symptoms
            if any(symptom in category_lower or symptom in symptoms.lower() 
                   for symptom in guide.get("symptoms", [])):
                matched_guide = guide
                break
        
        if matched_guide:
            response = {
                "success": True,
                "category": matched_guide["title"],
                "steps": matched_guide["steps"],
                "common_solutions": matched_guide.get("common_solutions", {}),
                "additional_info": matched_guide.get("password_policy") if "password" in category_lower else None
            }
            
            # Add context-specific recommendations if available
            if context:
                response["knowledge_base_info"] = context
                response["context_applied"] = True
            
            response["note"] = "💡 If these steps don't resolve your issue, I can create a support ticket for you."
            return response
        
        # Generic troubleshooting
        return {
            "success": True,
            "category": "General Troubleshooting",
            "steps": [
                {"step": 1, "action": "Restart your computer", "detail": "A fresh restart often resolves temporary issues"},
                {"step": 2, "action": "Check for updates", "detail": "Ensure your system and applications are up to date"},
                {"step": 3, "action": "Clear temporary files", "detail": "Run Disk Cleanup to remove temp files"},
                {"step": 4, "action": "Check system resources", "detail": "Open Task Manager to check CPU, memory usage"},
                {"step": 5, "action": "Document the error", "detail": "Take screenshots of any error messages"},
                {"step": 6, "action": "Create a support ticket", "detail": "If issue persists, I can help create a ticket"}
            ],
            "note": "For more specific troubleshooting, please describe your issue in more detail."
        }
    
    def request_software(
        self,
        user_id: str,
        software_name: str,
        justification: str = "",
        context: str = ""
    ) -> Dict[str, Any]:
        """
        Request software installation with smart catalog matching.
        """
        software_key = software_name.lower().replace(" ", "_").replace("-", "_")
        
        # Search across all categories
        for category, software_list in self.software_catalog.items():
            for key, info in software_list.items():
                if key in software_key or software_key in key or software_key in info["name"].lower():
                    if info["approval_required"]:
                        ticket = self.create_ticket(
                            user_id=user_id,
                            issue=f"Software Request: {info['name']}",
                            category="software_request",
                            priority="low",
                            description=f"Business justification: {justification}",
                            context=context
                        )
                        return {
                            "success": True,
                            "status": "pending_approval",
                            "software": info["name"],
                            "category": category,
                            "license_type": info["license"],
                            "ticket_id": ticket["ticket"]["id"],
                            "message": f"📋 {info['name']} requires manager approval.\n"
                                      f"Ticket {ticket['ticket']['id']} has been created.\n"
                                      f"Estimated install time once approved: {info['install_time']}",
                            "approval_process": [
                                "1. Your manager will receive an approval request",
                                "2. License availability will be verified",
                                "3. Once approved, IT will schedule installation",
                                "4. You'll receive a calendar invite for the installation"
                            ]
                        }
                    else:
                        return {
                            "success": True,
                            "status": "auto_approved",
                            "software": info["name"],
                            "category": category,
                            "license_type": info["license"],
                            "install_time": info["install_time"],
                            "message": f"✅ {info['name']} is pre-approved and can be installed immediately.\n"
                                      f"Estimated installation time: {info['install_time']}",
                            "installation_options": [
                                "1. Self-service: Install from Software Center",
                                "2. Remote install: IT can push the installation",
                                "3. Scheduled: Book a time slot with IT support"
                            ]
                        }
        
        # Software not in catalog
        ticket = self.create_ticket(
            user_id=user_id,
            issue=f"Software Request: {software_name}",
            category="software_request",
            priority="low",
            description=f"Software not in standard catalog.\nBusiness justification: {justification}",
            context=context
        )
        return {
            "success": True,
            "status": "review_required",
            "software": software_name,
            "ticket_id": ticket["ticket"]["id"],
            "message": f"⚠️ {software_name} is not in the standard software catalog.\n"
                      f"Ticket {ticket['ticket']['id']} created for IT review.\n"
                      "The IT team will evaluate compatibility, security, and licensing requirements.",
            "review_process": [
                "1. Security assessment will be performed",
                "2. Compatibility with company systems verified",
                "3. Licensing options will be evaluated",
                "4. You'll be notified of the decision within 3-5 business days"
            ]
        }
    
    def initiate_password_reset(
        self,
        user_id: str,
        target_system: str = "AD",
        reason: str = ""
    ) -> Dict[str, Any]:
        """
        Initiate password reset with security verification.
        """
        reset_token = str(uuid.uuid4())[:8].upper()
        expiry_time = datetime.now() + timedelta(minutes=30)
        
        system_info = {
            "AD": {"name": "Active Directory (Windows Login)", "affects": ["Windows login", "Network drives", "Intranet"]},
            "email": {"name": "Email Account", "affects": ["Outlook", "Webmail", "Mobile email"]},
            "vpn": {"name": "VPN Access", "affects": ["Remote access", "VPN client"]},
            "application": {"name": "Application-specific", "affects": ["Specified application only"]}
        }
        
        sys_details = system_info.get(target_system.upper(), system_info["AD"])
        
        return {
            "success": True,
            "reset_initiated": True,
            "reset_token": reset_token,
            "token_expiry": expiry_time.isoformat(),
            "target_system": sys_details["name"],
            "affected_services": sys_details["affects"],
            "instructions": [
                f"1. A password reset link has been sent to your registered email",
                f"2. Reset token: **{reset_token}** (valid for 30 minutes)",
                "3. Click the link or enter the token on the password reset page",
                "4. Create a new password following the requirements below"
            ],
            "password_requirements": {
                "minimum_length": 12,
                "must_contain": ["Uppercase letter", "Lowercase letter", "Number", "Special character (!@#$%^&*)"],
                "restrictions": ["Cannot reuse last 10 passwords", "Cannot contain your username", "Cannot contain common words"]
            },
            "security_note": "⚠️ If you didn't request this reset, please contact IT Security immediately.",
            "message": f"🔐 Password reset initiated for {sys_details['name']}. Check your email for the reset link."
        }
    
    def check_system_status(self, system: str = None) -> Dict[str, Any]:
        """Check current status of IT systems"""
        if system:
            system_lower = system.lower().replace(" ", "_")
            for key, status in self.system_status.items():
                if key in system_lower or system_lower in key:
                    return {
                        "success": True,
                        "system": key,
                        "status": status["status"],
                        "last_checked": status["last_check"].isoformat(),
                        "message": f"{'✅' if status['status'] == 'operational' else '⚠️'} {key}: {status['status']}"
                    }
            return {"success": False, "message": f"System '{system}' not found in monitoring"}
        
        # Return all systems
        return {
            "success": True,
            "systems": {k: {"status": v["status"], "icon": "✅" if v["status"] == "operational" else "⚠️"} 
                       for k, v in self.system_status.items()},
            "overall_status": "All Systems Operational" if all(s["status"] == "operational" for s in self.system_status.values()) else "Some Systems Degraded",
            "last_updated": datetime.now().isoformat()
        }
    
    def escalate_ticket(self, ticket_id: str, reason: str) -> Dict[str, Any]:
        """Escalate a ticket to higher support level"""
        if ticket_id not in self.tickets:
            return {"success": False, "message": f"Ticket {ticket_id} not found"}
        
        ticket = self.tickets[ticket_id]
        ticket["escalation_level"] += 1
        ticket["notes"].append({
            "timestamp": datetime.now().isoformat(),
            "type": "escalation",
            "content": f"Escalated to Level {ticket['escalation_level']}. Reason: {reason}"
        })
        ticket["updated_at"] = datetime.now().isoformat()
        
        escalation_teams = {1: "Senior IT Support", 2: "IT Specialists", 3: "IT Management"}
        assigned_team = escalation_teams.get(ticket["escalation_level"], "IT Leadership")
        
        return {
            "success": True,
            "ticket_id": ticket_id,
            "new_escalation_level": ticket["escalation_level"],
            "assigned_to": assigned_team,
            "message": f"⬆️ Ticket {ticket_id} has been escalated to {assigned_team}. They will contact you within 2 hours."
        }
    
    def add_ticket_note(self, ticket_id: str, note: str, note_type: str = "update") -> Dict[str, Any]:
        """Add a note to an existing ticket"""
        if ticket_id not in self.tickets:
            return {"success": False, "message": f"Ticket {ticket_id} not found"}
        
        ticket = self.tickets[ticket_id]
        ticket["notes"].append({
            "timestamp": datetime.now().isoformat(),
            "type": note_type,
            "content": note
        })
        ticket["updated_at"] = datetime.now().isoformat()
        
        return {
            "success": True,
            "ticket_id": ticket_id,
            "message": f"📝 Note added to ticket {ticket_id}"
        }
    
    def resolve_ticket(self, ticket_id: str, resolution: str) -> Dict[str, Any]:
        """Mark a ticket as resolved"""
        if ticket_id not in self.tickets:
            return {"success": False, "message": f"Ticket {ticket_id} not found"}
        
        ticket = self.tickets[ticket_id]
        ticket["status"] = "Resolved"
        ticket["resolution"] = resolution
        ticket["resolved_at"] = datetime.now().isoformat()
        ticket["updated_at"] = datetime.now().isoformat()
        
        return {
            "success": True,
            "ticket_id": ticket_id,
            "resolution": resolution,
            "message": f"✅ Ticket {ticket_id} has been resolved"
        }
    
    def get_knowledge_article(self, topic: str, context: str = "") -> Dict[str, Any]:
        """
        Get knowledge base article with RAG-enhanced context.
        Integrates retrieved document context for richer responses.
        """
        # Search in troubleshooting guides
        topic_lower = topic.lower()
        
        for key, guide in self.troubleshooting_guides.items():
            if key in topic_lower or any(s in topic_lower for s in guide.get("symptoms", [])):
                article = {
                    "success": True,
                    "topic": guide["title"],
                    "content": guide,
                    "source": "IT Knowledge Base"
                }
                
                # Enhance with RAG context if available
                if context:
                    article["enhanced_context"] = context
                    article["context_source"] = "Document Knowledge Base"
                
                return article
        
        # If no internal match, rely on RAG context
        if context:
            return {
                "success": True,
                "topic": topic,
                "content": None,
                "enhanced_context": context,
                "context_source": "Document Knowledge Base",
                "note": "Information retrieved from document knowledge base"
            }
        
        return {
            "success": False,
            "message": f"No knowledge article found for '{topic}'. Would you like me to create a support ticket?"
        }
    
    def _auto_categorize(self, text: str) -> str:
        """Auto-categorize based on text content"""
        text_lower = text.lower()
        categories = {
            "network": ["network", "internet", "wifi", "vpn", "connection", "connectivity", "dns"],
            "email": ["email", "outlook", "mail", "inbox", "sending", "receiving", "exchange"],
            "software": ["software", "application", "app", "install", "crash", "error", "not working", "update"],
            "hardware": ["computer", "laptop", "monitor", "keyboard", "mouse", "printer", "slow", "screen"],
            "access": ["password", "login", "access", "permission", "locked", "account", "reset"],
            "security": ["virus", "malware", "phishing", "suspicious", "security", "antivirus"]
        }
        
        for category, keywords in categories.items():
            if any(kw in text_lower for kw in keywords):
                return category
        return "general"
    
    def _auto_assign(self, category: str, priority: str) -> str:
        """Auto-assign ticket based on category and priority"""
        assignments = {
            "network": "Network Team",
            "email": "Messaging Team",
            "software": "Desktop Support",
            "hardware": "Hardware Team",
            "access": "Identity & Access Team",
            "security": "Security Operations",
            "general": "IT Helpdesk"
        }
        
        if priority in ["critical", "high"]:
            return f"Senior {assignments.get(category, 'IT Support')}"
        return assignments.get(category, "IT Helpdesk")
    
    def _get_initial_recommendations(self, category: str, issue: str) -> List[str]:
        """Get initial recommendations based on category"""
        if category in self.troubleshooting_guides:
            guide = self.troubleshooting_guides[category]
            steps = guide.get("steps", [])
            if steps:
                first_step = steps[0]
                return [f"While waiting: {first_step.get('action', '')} - {first_step.get('detail', '')}"]
        return ["While waiting: Gather any error messages or screenshots that might help diagnose the issue."]
    
    def _calculate_sla(self, priority: str) -> str:
        """Calculate SLA based on priority"""
        return f"{self.priority_config.get(priority, self.priority_config['medium'])['sla_hours']} hours"
    
    def execute_action(self, action: str, parameters: Dict) -> Dict[str, Any]:
        """
        Execute a specific IT Service Desk action.
        
        This is the main entry point for the agent to interact with the tool.
        
        Args:
            action: Action to perform
            parameters: Action parameters including context from RAG
            
        Returns:
            Result dictionary with success status and relevant data
        """
        # Extract context if provided (from RAG retrieval)
        context = parameters.pop("context", "")
        
        actions = {
            "create_ticket": lambda **p: self.create_ticket(
                user_id=p.get("user_id", "user"),
                issue=p.get("issue", p.get("title", "Support Request")),
                category=p.get("category", "general"),
                priority=p.get("priority", "medium"),
                description=p.get("description", ""),
                context=context
            ),
            "check_status": lambda **p: self.get_ticket_status(
                ticket_id=p.get("ticket_id"),
                user_id=p.get("user_id")
            ),
            "password_reset": lambda **p: self.initiate_password_reset(
                user_id=p.get("user_id", "user"),
                target_system=p.get("target_system", p.get("system", "AD")),
                reason=p.get("reason", "")
            ),
            "software_request": lambda **p: self.request_software(
                user_id=p.get("user_id", "user"),
                software_name=p.get("software_name", p.get("software", "")),
                justification=p.get("justification", ""),
                context=context
            ),
            "troubleshoot": lambda **p: self.troubleshoot(
                category=p.get("category", p.get("issue_type", "general")),
                symptoms=p.get("symptoms", p.get("issue", "")),
                context=context
            ),
            "system_status": lambda **p: self.check_system_status(p.get("system")),
            "escalate": lambda **p: self.escalate_ticket(
                ticket_id=p.get("ticket_id", ""),
                reason=p.get("reason", "User requested escalation")
            ),
            "add_note": lambda **p: self.add_ticket_note(
                ticket_id=p.get("ticket_id", ""),
                note=p.get("note", ""),
                note_type=p.get("note_type", "update")
            ),
            "resolve_ticket": lambda **p: self.resolve_ticket(
                ticket_id=p.get("ticket_id", ""),
                resolution=p.get("resolution", "")
            ),
            "knowledge": lambda **p: self.get_knowledge_article(
                topic=p.get("topic", ""),
                context=context
            )
        }
        
        if action in actions:
            try:
                return actions[action](**parameters)
            except Exception as e:
                return {"success": False, "error": f"Action failed: {str(e)}"}
        
        return {
            "success": False, 
            "error": f"Unknown action: {action}",
            "available_actions": list(actions.keys())
        }
    
    def get_available_actions(self) -> Dict[str, str]:
        """Return list of available actions with descriptions"""
        return {
            "create_ticket": "Create a new IT support ticket for tracking and resolution",
            "check_status": "Check the status of an existing ticket or view all your tickets",
            "password_reset": "Initiate password reset for Windows, email, VPN, or other systems",
            "software_request": "Request installation of new software or applications",
            "troubleshoot": "Get step-by-step troubleshooting guidance for common IT issues",
            "system_status": "Check the current operational status of IT systems",
            "escalate": "Escalate an existing ticket to a higher support level",
            "add_note": "Add a note or update to an existing ticket",
            "resolve_ticket": "Mark a ticket as resolved with resolution details",
            "knowledge": "Search the IT knowledge base for helpful articles"
        }
