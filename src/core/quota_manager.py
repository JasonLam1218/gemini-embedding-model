import time
import json
from pathlib import Path
from datetime import datetime, timedelta
from loguru import logger

class APIQuotaManager:
    def __init__(self, quota_file: str = "data/output/api_quota.json"):
        self.quota_file = Path(quota_file)
        self.daily_limit = 50  # Gemini free tier limit
        self.load_quota_data()
    
    def load_quota_data(self):
        """Load quota usage data"""
        try:
            if self.quota_file.exists():
                with open(self.quota_file, 'r') as f:
                    self.quota_data = json.load(f)
            else:
                self.quota_data = {
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "requests_made": 0,
                    "requests_remaining": self.daily_limit
                }
                self.save_quota_data()
        except Exception as e:
            logger.error(f"Failed to load quota data: {e}")
            self.reset_quota_data()
    
    def save_quota_data(self):
        """Save quota usage data"""
        try:
            self.quota_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.quota_file, 'w') as f:
                json.dump(self.quota_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save quota data: {e}")
    
    def reset_quota_data(self):
        """Reset quota data for new day"""
        self.quota_data = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "requests_made": 0,
            "requests_remaining": self.daily_limit
        }
        self.save_quota_data()
    
    def check_and_update_quota(self) -> bool:
        """Check if request can be made and update quota"""
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        # Reset if new day
        if self.quota_data["date"] != current_date:
            self.reset_quota_data()
        
        # Check if quota available
        if self.quota_data["requests_remaining"] <= 0:
            logger.warning(f"⚠️ API quota exhausted for today ({self.quota_data['requests_made']}/{self.daily_limit})")
            return False
        
        # Update quota
        self.quota_data["requests_made"] += 1
        self.quota_data["requests_remaining"] -= 1
        self.save_quota_data()
        
        return True
    
    def get_quota_status(self) -> dict:
        """Get current quota status"""
        return {
            "date": self.quota_data["date"],
            "requests_made": self.quota_data["requests_made"],
            "requests_remaining": self.quota_data["requests_remaining"],
            "daily_limit": self.daily_limit
        }
