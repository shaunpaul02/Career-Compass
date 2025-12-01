import logging
import json
from typing import Dict, Any
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def get_logger(name: str) -> logging.Logger:
    """Get a configured logger instance"""
    return logging.getLogger(name)

def format_trait_object(traits: Dict[str, Any]) -> str:
    """Format trait object for display"""
    return json.dumps(traits, indent=2)

def calculate_compatibility_score(user_traits: Dict, job_traits: Dict) -> float:
    """
    Calculate compatibility score between user traits and job requirements.
    Returns a score between 0.0 and 1.0
    """
    if not user_traits or not job_traits:
        return 0.0

    matching_traits = 0
    total_traits = len(job_traits)

    for trait, required_level in job_traits.items():
        if trait in user_traits:
            user_level = user_traits[trait]
            # Simple matching: if levels match or user exceeds requirement
            if _compare_levels(user_level, required_level):
                matching_traits += 1

    return min(1.0, matching_traits / total_traits) if total_traits > 0 else 0.5

def _compare_levels(user_level: Any, required_level: Any) -> bool:
    """Compare trait levels"""
    level_order = {"low": 1, "medium": 2, "high": 3}

    # Convert to comparable format
    u_val = level_order.get(str(user_level).lower(), 2)
    r_val = level_order.get(str(required_level).lower(), 2)

    return u_val >= r_val

def parse_job_description(job_desc: str) -> Dict[str, Any]:
    """
    Parse job description and extract key requirements.
    Returns a dict with extracted traits and requirements.
    """
    traits = {
        "technical_skills": [],
        "soft_skills": [],
        "experience_level": "mid",
        "work_environment": "standard"
    }

    desc_lower = job_desc.lower()

    # Detect environment type
    if "fast-paced" in desc_lower or "high-pressure" in desc_lower:
        traits["work_environment"] = "fast_paced"
    elif "remote" in desc_lower or "flexible" in desc_lower:
        traits["work_environment"] = "flexible"

    # Detect experience level
    if "senior" in desc_lower or "lead" in desc_lower:
        traits["experience_level"] = "senior"
    elif "entry" in desc_lower or "junior" in desc_lower:
        traits["experience_level"] = "entry"

    return traits

def timestamp() -> str:
    """Return current timestamp"""
    return datetime.now().isoformat()
