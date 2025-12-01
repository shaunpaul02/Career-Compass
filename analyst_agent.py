"""
Analyst Agent: Responsible for evaluating job matches against user profile.
Demonstrates data synthesis and scoring logic.
"""

from typing import Dict, Any, List, Tuple
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from utils import get_logger, timestamp, calculate_compatibility_score
import json
import config

logger = get_logger(__name__)


class MatchCard(BaseModel):
    """Detailed match analysis for a specific job"""
    job_title: str
    company: str
    compatibility_score: float = Field(ge=0.0, le=1.0)
    match_percentage: int  # 0-100%
    reasoning: str
    matched_traits: List[str]
    unmatched_traits: List[str]
    recommendation: str  # "Strong Match", "Good Fit", "Consider", "Not Aligned"
    market_availability: str  # Based on applicant count
    key_strengths: List[str]
    growth_areas: List[str]


class AnalystAgent:
    """
    The Analyst Agent evaluates jobs against user profiles.

    Demonstrates:
    - Data Synthesis: Combining multiple data sources
    - Scoring Logic: Calculating compatibility scores
    - Reasoning Generation: Explaining why a job matches
    """

    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model=config.MODEL_NAME,
            temperature=config.TEMPERATURE,
            google_api_key=config.GOOGLE_API_KEY
        )
        self.analysis_history: List[Dict[str, Any]] = []
        logger.info("AnalystAgent initialized")

    def analyze_job_fit(self,
                       job: Dict[str, Any],
                       user_profile: Dict[str, Any]) -> MatchCard:
        """
        Analyze how well a job matches the user's profile.

        Args:
            job: Job posting from Scout Agent
            user_profile: User traits from Profiler Agent

        Returns:
            MatchCard: Detailed analysis and compatibility score
        """
        logger.info(f"Analyzing fit for: {job.get('title', 'Unknown')}")

        # Extract job traits from description
        job_traits = self._extract_job_traits(job)

        # Calculate base compatibility score
        user_traits_for_scoring = {
            "resilience": user_profile.get("resilience", "medium"),
            "leadership": user_profile.get("leadership", "medium"),
            "technical_aptitude": user_profile.get("technical_aptitude", "medium"),
            "problem_solving": user_profile.get("problem_solving", "medium"),
            "teamwork": user_profile.get("teamwork", "medium"),
            "communication": user_profile.get("communication", "medium"),
        }

        compatibility_score = calculate_compatibility_score(user_traits_for_scoring, job_traits)

        # Get AI-powered reasoning
        reasoning = self._generate_reasoning(job, user_profile, compatibility_score)

        # Determine matched and unmatched traits
        matched_traits, unmatched_traits = self._identify_trait_alignment(
            user_profile, job_traits
        )

        # Generate recommendation
        recommendation = self._generate_recommendation(compatibility_score)

        # Assess market availability
        market_availability = self._assess_market_availability(job)

        # Identify strengths and growth areas
        strengths = self._identify_strengths(matched_traits)
        growth_areas = self._identify_growth_areas(unmatched_traits)

        match_card = MatchCard(
            job_title=job.get("title", "Unknown"),
            company=job.get("company", "Unknown"),
            compatibility_score=compatibility_score,
            match_percentage=int(compatibility_score * 100),
            reasoning=reasoning,
            matched_traits=matched_traits,
            unmatched_traits=unmatched_traits,
            recommendation=recommendation,
            market_availability=market_availability,
            key_strengths=strengths,
            growth_areas=growth_areas
        )

        # Store in history
        self.analysis_history.append({
            "timestamp": timestamp(),
            "job_id": f"{job.get('company')}_{job.get('title')}",
            "match_card": match_card.dict()
        })

        logger.info(f"Analysis complete: {recommendation} ({match_card.match_percentage}% match)")
        return match_card

    def _extract_job_traits(self, job: Dict[str, Any]) -> Dict[str, str]:
        """
        Extract required traits from job description using keyword analysis.

        Demonstrates how the agent interprets unstructured job data.
        """
        description = job.get("description", "").lower()
        title = job.get("title", "").lower()
        combined_text = f"{title} {description}"

        traits = {
            "resilience": "medium",
            "leadership": "medium",
            "technical_aptitude": "medium",
            "problem_solving": "medium",
            "teamwork": "medium",
            "communication": "medium",
        }

        # Keyword-based extraction
        if any(word in combined_text for word in ["lead", "manage", "team", "director", "manager", "head"]):
            traits["leadership"] = "high"

        if any(word in combined_text for word in ["develop", "code", "technical", "engineer", "architect", "software"]):
            traits["technical_aptitude"] = "high"

        if any(word in combined_text for word in ["analyze", "solve", "problem", "critical", "strategic", "optimize"]):
            traits["problem_solving"] = "high"

        if any(word in combined_text for word in ["collaborate", "team", "group", "together", "communication"]):
            traits["teamwork"] = "high"

        if any(word in combined_text for word in ["communicate", "present", "speak", "client", "stakeholder"]):
            traits["communication"] = "high"

        if any(word in combined_text for word in ["pressure", "fast-paced", "emergency", "urgent", "crisis", "resilient"]):
            traits["resilience"] = "high"

        return traits

    def _identify_trait_alignment(self,
                                   user_profile: Dict[str, Any],
                                   job_traits: Dict[str, str]) -> Tuple[List[str], List[str]]:
        """
        Identify which user traits match job requirements and which don't.
        """
        matched = []
        unmatched = []

        for trait in job_traits:
            user_level = user_profile.get(trait, "medium")
            job_level = job_traits.get(trait, "medium")

            # Convert levels to numeric for comparison
            level_map = {"low": 1, "medium": 2, "high": 3}
            user_val = level_map.get(user_level, 2)
            job_val = level_map.get(job_level, 2)

            if user_val >= job_val:
                matched.append(f"Strong {trait.replace('_', ' ').title()}")
            elif user_val == job_val - 1:
                matched.append(f"Good {trait.replace('_', ' ').title()}")
            else:
                unmatched.append(f"Developing {trait.replace('_', ' ').title()}")

        return matched[:5], unmatched[:3]  # Limit to top 5 and 3

    def _generate_recommendation(self, compatibility_score: float) -> str:
        """Generate recommendation based on compatibility score"""
        if compatibility_score >= 0.8:
            return "Strong Match"
        elif compatibility_score >= 0.6:
            return "Good Fit"
        elif compatibility_score >= 0.4:
            return "Consider"
        else:
            return "Not Aligned"

    def _generate_reasoning(self,
                           job: Dict[str, Any],
                           user_profile: Dict[str, Any],
                           score: float) -> str:
        """
        Generate AI-powered reasoning for why this job matches (or doesn't).
        This demonstrates the Analyst's synthesis capability.
        """
        job_title = job.get("title", "this role")
        company = job.get("company", "this company")
        description = job.get("description", "")

        # Build reasoning narrative
        reasoning_parts = []

        # Start with overall assessment
        recommendation = self._generate_recommendation(score)
        reasoning_parts.append(f"{score*100:.0f}% Match - {recommendation}")

        # Add specific observations
        if "problem_solving" in user_profile and user_profile.get("problem_solving") == "high":
            if "problem" in description.lower() or "analyze" in description.lower():
                reasoning_parts.append(
                    "Your strong problem-solving skills align well with this role's analytical demands."
                )

        if "resilience" in user_profile and user_profile.get("resilience") == "high":
            if "fast-paced" in description.lower() or "pressure" in description.lower():
                reasoning_parts.append(
                    "This fast-paced environment matches your ability to thrive under pressure."
                )

        if "leadership" in user_profile and user_profile.get("leadership") == "high":
            if any(word in job_title.lower() for word in ["manager", "lead", "director"]):
                reasoning_parts.append(
                    "Your leadership capabilities are well-suited for this management track."
                )

        if not reasoning_parts or len(reasoning_parts) == 1:
            # Provide generic reasoning if specific traits don't match
            reasoning_parts.append(
                f"The overall skill profile and work style preferences align with what {company} is looking for in this {job_title} position."
            )

        return " ".join(reasoning_parts)

    def _assess_market_availability(self, job: Dict[str, Any]) -> str:
        """
        Assess job market availability based on applicant count.
        Helps user prioritize where effort is best spent.
        """
        applicants = job.get("applicants", 100)

        if applicants < 30:
            return "High Availability (Few Applicants)"
        elif applicants < 100:
            return "Moderate Availability"
        elif applicants < 300:
            return "Competitive Market"
        else:
            return "Highly Competitive (Many Applicants)"

    def _identify_strengths(self, matched_traits: List[str]) -> List[str]:
        """Identify user's key strengths for this role"""
        return matched_traits[:3] if matched_traits else ["Potential for role growth"]

    def _identify_growth_areas(self, unmatched_traits: List[str]) -> List[str]:
        """Identify areas where user could develop further"""
        if not unmatched_traits:
            return ["Continue skill development in technical areas"]
        return [f"Develop {trait}" for trait in unmatched_traits[:2]]

    def batch_analyze_jobs(self,
                          jobs: List[Dict[str, Any]],
                          user_profile: Dict[str, Any]) -> List[MatchCard]:
        """
        Analyze multiple jobs at once and rank them.

        Demonstrates the agent's ability to synthesize and rank data.
        """
        match_cards = []

        for job in jobs:
            try:
                match_card = self.analyze_job_fit(job, user_profile)
                match_cards.append(match_card)
            except Exception as e:
                logger.error(f"Error analyzing job {job.get('title')}: {str(e)}")

        # Sort by compatibility score (descending)
        match_cards.sort(key=lambda x: x.compatibility_score, reverse=True)

        logger.info(f"Ranked {len(match_cards)} jobs")
        return match_cards

    def get_top_matches(self,
                       jobs: List[Dict[str, Any]],
                       user_profile: Dict[str, Any],
                       top_n: int = 5) -> List[MatchCard]:
        """
        Get top N job matches.
        """
        all_matches = self.batch_analyze_jobs(jobs, user_profile)
        return all_matches[:top_n]

    def get_analysis_history(self) -> List[Dict[str, Any]]:
        """Return analysis history"""
        return self.analysis_history.copy()

    def reset(self) -> None:
        """Reset analyst for new session"""
        self.analysis_history = []
        logger.info("AnalystAgent reset")
