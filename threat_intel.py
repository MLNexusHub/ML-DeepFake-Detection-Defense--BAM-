import logging
import random
import json
import os
import requests
import time
from datetime import datetime, timedelta
from pathlib import Path
import dotenv
from urllib.parse import urljoin

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
dotenv.load_dotenv()

class ThreatIntelligence:
    """Implementation of ThreatIntelligence with URLScan.io API support."""
    
    def __init__(self, api_config=None, update_interval=3600):
        """
        Initialize the threat intelligence module.
        
        Args:
            api_config: Configuration for APIs (optional)
            update_interval: How often to update threat data in seconds (default: 1 hour)
        """
        logger.info("Initializing ThreatIntelligence")
        self.api_config = api_config or {}
        self.update_interval = update_interval
        
        # Get API key from environment variables if not in config
        self.urlscan_api_key = os.getenv("URLSCAN_API_KEY")
        if not self.urlscan_api_key and 'threat_intel_apis' in self.api_config:
            if 'urlscan' in self.api_config['threat_intel_apis']:
                self.urlscan_api_key = self.api_config['threat_intel_apis']['urlscan'].get('key')
        
        # Check if we have a valid API key
        self.use_mock = not self.urlscan_api_key
        if self.use_mock:
            logger.warning("No URLScan.io API key found. Using mock data.")
        else:
            logger.info("URLScan.io API key found. Using real API.")
        
        # Setup local storage for threat data
        self._setup_local_storage()
        
        # Initialize threat data
        if self.use_mock:
            # Generate initial mock threats
            self._mock_threats = self._get_mock_threats(30)  # Generate 30 mock threats
            self._threat_data = self._mock_threats
        else:
            # Initialize with empty data, will be populated on first call
            self._threat_data = []
            
        self._last_update = datetime.now()
        
        # Cache for threat statistics
        self._threat_stats = self._calculate_threat_stats()
        
        logger.info(f"ThreatIntelligence initialized with {'mock' if self.use_mock else 'real'} data")
    
    def _setup_local_storage(self):
        """Setup local storage for threat data."""
        # Create directory for threat data if it doesn't exist
        self.storage_path = Path("detection_results/threats")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Path to threat data file
        self.threats_file = self.storage_path / "threats.json"
        
        # Initialize threats file if it doesn't exist
        if not self.threats_file.exists():
            with open(self.threats_file, 'w') as f:
                json.dump([], f)
    
    def _get_mock_threats(self, count=10):
        """
        Generate mock threat data for testing.
        
        Args:
            count: Number of mock threats to generate
            
        Returns:
            list: List of mock threat dictionaries
        """
        threat_types = [
            "deepfake_campaign", "brand_impersonation", "credential_theft",
            "phishing", "social_engineering", "fake_news", "disinformation"
        ]
        
        sources = [
            "urlscan.io", "phishtank", "internal_analysis", 
            "user_report", "partner_feed", "osint"
        ]
        
        threats = []
        now = datetime.now()
        
        for i in range(count):
            # Generate random timestamp within the last 7 days
            days_ago = random.uniform(0, 7)
            timestamp = now - timedelta(days=days_ago)
            
            # Generate random severity (weighted towards medium)
            severity_weights = [0.2, 0.5, 0.3]  # low, medium, high
            severity = random.choices(["low", "medium", "high"], weights=severity_weights)[0]
            
            # Generate random threat type
            threat_type = random.choice(threat_types)
            
            # Generate random source
            source = random.choice(sources)
            
            # Generate random indicators based on threat type
            indicators = self._generate_mock_indicators(threat_type)
            
            # Create threat entry
            threat = {
                "id": f"THREAT-{int(time.time())}-{i}",
                "type": threat_type,
                "severity": severity,
                "description": self._generate_description(threat_type, indicators),
                "timestamp": timestamp.isoformat(),
                "source": source,
                "indicators": indicators,
                "status": random.choice(["active", "active", "active", "mitigated"])  # Bias towards active
            }
            
            threats.append(threat)
        
        return threats
    
    def _generate_mock_indicators(self, threat_type):
        """Generate mock indicators based on threat type."""
        indicators = {}
        
        if threat_type in ["deepfake_campaign", "fake_news", "disinformation"]:
            indicators["domains"] = [
                f"fake-{random.randint(100, 999)}.com",
                f"news-{random.randint(100, 999)}.org"
            ]
            indicators["social_accounts"] = [
                f"@fake_account_{random.randint(1000, 9999)}",
                f"@news_spreader_{random.randint(1000, 9999)}"
            ]
            
        if threat_type in ["brand_impersonation", "credential_theft", "phishing"]:
            indicators["domains"] = [
                f"secure-{random.randint(100, 999)}.com",
                f"login-{random.randint(100, 999)}.net"
            ]
            indicators["urls"] = [
                f"https://secure-{random.randint(100, 999)}.com/login",
                f"https://login-{random.randint(100, 999)}.net/reset"
            ]
            indicators["emails"] = [
                f"security@secure-{random.randint(100, 999)}.com",
                f"support@login-{random.randint(100, 999)}.net"
            ]
            
        if threat_type in ["social_engineering"]:
            indicators["keywords"] = [
                "urgent", "action required", "security alert",
                "password reset", "account verification"
            ]
            indicators["phone_numbers"] = [
                f"+1-{random.randint(100, 999)}-{random.randint(100, 999)}-{random.randint(1000, 9999)}",
                f"+44-{random.randint(100, 999)}-{random.randint(1000, 9999)}"
            ]
        
        return indicators
    
    def _generate_description(self, threat_type, indicators):
        """Generate a description based on threat type and indicators."""
        descriptions = {
            "deepfake_campaign": f"Deepfake campaign detected using {len(indicators.get('domains', []))} domains to spread manipulated media.",
            "brand_impersonation": f"Brand impersonation attack targeting users via {len(indicators.get('urls', []))} phishing URLs.",
            "credential_theft": f"Credential theft campaign using {len(indicators.get('emails', []))} sender addresses.",
            "phishing": f"Phishing campaign detected across {len(indicators.get('domains', []))} domains.",
            "social_engineering": f"Social engineering attack using {len(indicators.get('keywords', []))} trigger keywords.",
            "fake_news": f"Fake news campaign spreading across {len(indicators.get('social_accounts', []))} social media accounts.",
            "disinformation": f"Disinformation campaign detected using {len(indicators.get('domains', []))} websites."
        }
        
        return descriptions.get(threat_type, f"Unknown threat type: {threat_type}")
    
    def _fetch_urlscan_threats(self):
        """
        Fetch threat data from URLScan.io API.
        
        Returns:
            list: List of threat dictionaries from URLScan.io
        """
        if not self.urlscan_api_key:
            logger.warning("No URLScan.io API key available. Cannot fetch real threats.")
            return []
            
        try:
            # URLScan.io API endpoint for searching
            base_url = "https://urlscan.io/api/v1/"
            search_endpoint = "search/"
            
            # Search for malicious URLs in the last 7 days
            # Focus on phishing, malware, and other threats
            query = "task.tags:malicious OR page.tags:malicious OR task.tags:phishing OR page.tags:phishing"
            
            # Set up request headers with API key
            headers = {
                "API-Key": self.urlscan_api_key,
                "Content-Type": "application/json"
            }
            
            # Make the request
            response = requests.get(
                urljoin(base_url, search_endpoint),
                params={"q": query, "size": 100},
                headers=headers,
                timeout=10
            )
            
            # Check if request was successful
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                
                # Transform URLScan.io data to our threat format
                threats = []
                for result in results:
                    # Extract relevant information
                    task = result.get("task", {})
                    page = result.get("page", {})
                    stats = result.get("stats", {})
                    
                    # Determine severity based on tags and stats
                    severity = "medium"  # Default
                    tags = task.get("tags", []) + page.get("tags", [])
                    
                    if "malware" in tags or stats.get("malicious", 0) > 5:
                        severity = "high"
                    elif "phishing" in tags or "scam" in tags:
                        severity = "high"
                    elif stats.get("suspicious", 0) > 3:
                        severity = "medium"
                    else:
                        severity = "low"
                    
                    # Determine threat type
                    threat_type = "phishing"  # Default
                    if "malware" in tags:
                        threat_type = "malware"
                    elif "phishing" in tags:
                        threat_type = "phishing"
                    elif "scam" in tags:
                        threat_type = "brand_impersonation"
                    elif "fake" in tags:
                        threat_type = "fake_news"
                    
                    # Create indicators
                    indicators = {
                        "domains": [page.get("domain", "unknown")],
                        "urls": [task.get("url", "unknown")],
                        "ips": [page.get("ip", "unknown")]
                    }
                    
                    # Create threat entry
                    threat = {
                        "id": result.get("_id", f"URLSCAN-{int(time.time())}"),
                        "type": threat_type,
                        "severity": severity,
                        "description": f"Threat detected: {page.get('domain', 'unknown')} - {task.get('url', 'unknown')}",
                        "timestamp": task.get("time", datetime.now().isoformat()),
                        "source": "urlscan.io",
                        "indicators": indicators,
                        "status": "active",
                        "raw_data": {
                            "score": result.get("score", 0),
                            "tags": tags,
                            "stats": stats
                        }
                    }
                    
                    threats.append(threat)
                
                logger.info(f"Successfully fetched {len(threats)} threats from URLScan.io")
                return threats
                
            else:
                logger.error(f"Failed to fetch threats from URLScan.io: {response.status_code} - {response.text}")
                return []
                
        except Exception as e:
            logger.error(f"Error fetching threats from URLScan.io: {str(e)}")
            return []
    
    def _calculate_threat_stats(self):
        """Calculate statistics from threat data."""
        stats = {
            "total_threats": len(self._threat_data),
            "active_threats": sum(1 for t in self._threat_data if t.get("status") == "active"),
            "mitigated_threats": sum(1 for t in self._threat_data if t.get("status") == "mitigated"),
            "high_severity": sum(1 for t in self._threat_data if t.get("severity") == "high"),
            "medium_severity": sum(1 for t in self._threat_data if t.get("severity") == "medium"),
            "low_severity": sum(1 for t in self._threat_data if t.get("severity") == "low"),
            "by_type": {},
            "by_source": {}
        }
        
        # Count by type
        for threat in self._threat_data:
            threat_type = threat.get("type", "unknown")
            source = threat.get("source", "unknown")
            
            if threat_type not in stats["by_type"]:
                stats["by_type"][threat_type] = 0
            stats["by_type"][threat_type] += 1
            
            if source not in stats["by_source"]:
                stats["by_source"][source] = 0
            stats["by_source"][source] += 1
        
        return stats
    
    def get_recent_threats(self, hours=24):
        """
        Get recent threats from the last specified hours.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            list: List of recent threats
        """
        # Check if we need to update threat data
        current_time = datetime.now()
        if (current_time - self._last_update).total_seconds() > self.update_interval:
            if self.use_mock:
                # Generate some new mock threats
                new_threats = self._get_mock_threats(5)  # Add 5 new threats
                self._mock_threats.extend(new_threats)
                self._threat_data = self._mock_threats
                logger.info(f"Generated {len(new_threats)} new mock threats")
            else:
                # Fetch real threats from URLScan.io
                real_threats = self._fetch_urlscan_threats()
                if real_threats:
                    self._threat_data = real_threats
                    # Save to local storage
                    self.save_threats()
                    logger.info(f"Updated threat data with {len(real_threats)} real threats from URLScan.io")
                
            self._last_update = current_time
            self._threat_stats = self._calculate_threat_stats()
        
        # Filter threats by time
        threshold = current_time - timedelta(hours=hours)
        recent_threats = []
        
        for threat in self._threat_data:
            try:
                threat_time = datetime.fromisoformat(threat.get("timestamp").replace("Z", "+00:00"))
                if threat_time > threshold:
                    recent_threats.append(threat)
            except (ValueError, TypeError, AttributeError) as e:
                # Skip threats with invalid timestamps
                logger.warning(f"Invalid timestamp in threat: {str(e)}")
                continue
        
        logger.info(f"Retrieved {len(recent_threats)} recent threats from the last {hours} hours")
        return recent_threats
    
    def get_threat_stats(self):
        """
        Get threat statistics.
        
        Returns:
            dict: Dictionary of threat statistics
        """
        return self._threat_stats
    
    def save_threats(self):
        """Save threats to local storage."""
        try:
            with open(self.threats_file, 'w') as f:
                json.dump(self._threat_data, f, indent=2)
            logger.info(f"Saved {len(self._threat_data)} threats to {self.threats_file}")
        except Exception as e:
            logger.error(f"Failed to save threats: {str(e)}")


def main():
    """Test the threat intelligence module."""
    # Initialize threat intelligence
    config = {
        'threat_intel_apis': {
            'urlscan': {
                'url': 'https://urlscan.io/api/v1/',
                'key': os.getenv("URLSCAN_API_KEY"),
                'enabled': True
            }
        },
        'development': {
            'use_mock_data': False,
            'mock_delay': 1,
            'save_local': True,
            'local_save_path': 'detection_results'
        }
    }
    
    threat_intel = ThreatIntelligence(api_config=config)
    
    # Get recent threats
    recent_threats = threat_intel.get_recent_threats(hours=48)
    print(f"Recent threats: {len(recent_threats)}")
    
    # Get threat stats
    stats = threat_intel.get_threat_stats()
    print("\nThreat Statistics:")
    print(f"Total Threats: {stats['total_threats']}")
    print(f"Active Threats: {stats['active_threats']}")
    print(f"High Severity: {stats['high_severity']}")
    print(f"Medium Severity: {stats['medium_severity']}")
    print(f"Low Severity: {stats['low_severity']}")
    
    print("\nThreats by Type:")
    for threat_type, count in stats['by_type'].items():
        print(f"  - {threat_type}: {count}")
    
    print("\nThreats by Source:")
    for source, count in stats['by_source'].items():
        print(f"  - {source}: {count}")
    
    # Save threats to local storage
    threat_intel.save_threats()


if __name__ == "__main__":
    main() 