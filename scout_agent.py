"""
Scout Agent: Responsible for searching and retrieving relevant job postings.
Uses tool calling to autonomously decide when and what to search for.
Demonstrates Function Calling concept.
"""

from typing import Dict, Any, List, Optional
import requests
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from utils import get_logger, timestamp
import config

logger = get_logger(__name__)


class SearchTool:
    """
    Custom Search Tool that wraps Google Custom Search API.
    Demonstrates the Tool/Function Calling concept.
    """

    def __init__(self, api_key: str = None, search_engine_id: str = None):
        self.api_key = api_key or config.GOOGLE_API_KEY
        self.search_engine_id = search_engine_id or config.GOOGLE_SEARCH_ENGINE_ID
        self.base_url = "https://www.googleapis.com/customsearch/v1"

    def search_jobs(self, 
                    job_title: str, 
                    location: str = None,
                    keywords: List[str] = None) -> List[Dict[str, Any]]:
        """
        Search for job postings based on criteria.

        Args:
            job_title: Job title to search for
            location: Geographic location (e.g., "London, ON")
            keywords: Additional keywords to include in search

        Returns:
            List of job posting dicts with title, description, link, etc.
        """
        location = location or config.DEFAULT_LOCATION
        logger.info(f"Searching for: {job_title} in {location}")

        # Build search query
        query_parts = [job_title, "job", location]
        if keywords:
            query_parts.extend(keywords[:3])  # Add up to 3 keywords

        search_query = " ".join(query_parts)

        try:
            # For demo purposes, using mock data since real API requires setup
            # In production, uncomment the real API call below

            # Real API call:
            # params = {
            #     "q": search_query,
            #     "key": self.api_key,
            #     "cx": self.search_engine_id,
            #     "num": config.MAX_SEARCH_RESULTS
            # }
            # response = requests.get(self.base_url, params=params, timeout=10)
            # response.raise_for_status()
            # results = response.json().get("items", [])

            # Mock results for testing
            results = self._get_mock_results(job_title, location)

            logger.info(f"Found {len(results)} results for '{search_query}'")
            return results

        except requests.exceptions.RequestException as e:
            logger.error(f"Search API error: {str(e)}")
            return []

    def _get_mock_results(self, job_title: str, location: str) -> List[Dict[str, Any]]:
        """
        Return mock job results for testing.
        In production, replace with real API results.
        """
        mock_jobs = {
            "project manager": [
                {
                    "title": "Senior Project Manager",
                    "company": "TechCorp Solutions",
                    "location": location,
                    "description": "Lead cross-functional teams in delivering software projects. Requires strong leadership, fast-paced environment.",
                    "link": "https://example.com/job1",
                    "applicants": 45,
                    "salary_range": "$80k - $120k"
                },
                {
                    "title": "Project Manager - Client Services",
                    "company": "Global Consulting",
                    "location": location,
                    "description": "Manage client projects with focus on communication and collaboration. Work in structured, professional environment.",
                    "link": "https://example.com/job2",
                    "applicants": 120,
                    "salary_range": "$70k - $100k"
                }
            ],
            "software engineer": [
                {
                    "title": "Full Stack Software Engineer",
                    "company": "InnovateTech",
                    "location": location,
                    "description": "Build scalable web applications. High-pressure, fast-paced startup environment. Strong problem-solving required.",
                    "link": "https://example.com/job3",
                    "applicants": 250,
                    "salary_range": "$75k - $110k"
                },
                {
                    "title": "Backend Engineer",
                    "company": "StableCorpServices",
                    "location": location,
                    "description": "Develop backend systems in a structured environment. Focus on code quality and testing.",
                    "link": "https://example.com/job4",
                    "applicants": 180,
                    "salary_range": "$80k - $115k"
                }
            ],
            "emergency dispatcher": [
                {
                    "title": "Emergency Communications Operator",
                    "company": "City Emergency Services",
                    "location": location,
                    "description": "Dispatch emergency responders. High-pressure environment requiring quick decision-making and stress resilience.",
                    "link": "https://example.com/job5",
                    "applicants": 35,
                    "salary_range": "$50k - $70k"
                },
                {
                    "title": "911 Dispatcher",
                    "company": "Regional Response Center",
                    "location": location,
                    "description": "Handle emergency calls and coordinate response teams. Requires resilience and calm under pressure.",
                    "link": "https://example.com/job6",
                    "applicants": 28,
                    "salary_range": "$48k - $68k"
                }
            ]
        }

        # Find matching category
        for category, jobs in mock_jobs.items():
            if category in job_title.lower():
                return jobs[:config.MAX_SEARCH_RESULTS]

        # Default: return generic jobs
        return mock_jobs.get("project manager", [])[:config.MAX_SEARCH_RESULTS]


class ScoutAgent:
    """
    The Scout Agent searches for job postings based on user traits.

    Demonstrates:
    - Function Calling: Autonomously deciding when and what to search
    - Tool Use: Integration with SearchTool for real-time data retrieval
    - Reasoning: Converting abstract traits into concrete search queries
    """

    def __init__(self, search_tool: Optional[SearchTool] = None):
        self.llm = ChatGoogleGenerativeAI(
            model=config.MODEL_NAME,
            temperature=config.TEMPERATURE,
            google_api_key=config.GOOGLE_API_KEY
        )
        self.search_tool = search_tool or SearchTool()
        self.search_history: List[Dict[str, Any]] = []
        logger.info("ScoutAgent initialized")

    def find_matching_jobs(self, 
                           user_profile: Dict[str, Any],
                           location: str = None) -> List[Dict[str, Any]]:
        """
        Find jobs matching user's trait profile.
        This demonstrates autonomous tool use: the agent decides WHEN to search
        based on having sufficient profile data.

        Args:
            user_profile: Dictionary containing user traits from ProfilerAgent
            location: Job location (default: London, ON)

        Returns:
            List of job postings with metadata
        """
        location = location or config.DEFAULT_LOCATION
        logger.info(f"Finding jobs for profile: {user_profile.get('work_style', 'unknown')}")

        if not self._has_sufficient_profile_data(user_profile):
            logger.warning("Insufficient profile data for job search")
            return []

        # Convert traits to job search queries
        search_queries = self._generate_search_queries(user_profile)
        logger.info(f"Generated search queries: {search_queries}")

        all_jobs = []

        for query in search_queries:
            logger.info(f"Executing search: {query}")

            search_record = {
                "timestamp": timestamp(),
                "query": query,
                "location": location,
                "status": "searching"
            }

            try:
                # Call the search tool
                jobs = self.search_tool.search_jobs(
                    job_title=query,
                    location=location,
                    keywords=user_profile.get("keywords", [])
                )

                search_record["status"] = "completed"
                search_record["results_count"] = len(jobs)
                all_jobs.extend(jobs)

            except Exception as e:
                logger.error(f"Search failed: {str(e)}")
                search_record["status"] = "failed"
                search_record["error"] = str(e)

            self.search_history.append(search_record)

        # Deduplicate jobs
        unique_jobs = self._deduplicate_jobs(all_jobs)
        logger.info(f"Found {len(unique_jobs)} unique job postings")

        return unique_jobs

    def _has_sufficient_profile_data(self, profile: Dict[str, Any]) -> bool:
        """
        Check if profile has enough data to make meaningful searches.
        Demonstrates autonomous decision-making.
        """
        required_fields = ["resilience", "problem_solving", "environment_preference"]
        return all(field in profile and profile[field] != "medium" 
                  for field in required_fields) or len(profile.get("keywords", [])) > 0

    def _generate_search_queries(self, profile: Dict[str, Any]) -> List[str]:
        """
        Convert trait profile into search job titles and keywords.

        Demonstrates reasoning: abstract traits -> concrete job searches
        """
        queries = []

        # Map traits to job roles
        trait_to_jobs = {
            ("high", "leadership"): ["Project Manager", "Team Lead", "Manager"],
            ("high", "technical_aptitude"): ["Software Engineer", "Developer", "Architect"],
            ("high", "problem_solving"): ["Data Analyst", "Business Analyst", "Engineer"],
            ("high", "creativity"): ["Designer", "Creative Director", "Product Manager"],
            ("high", "resilience"): ["Emergency Dispatcher", "Sales Manager", "Operations Manager"],
        }

        # Primary queries from direct trait mapping
        for trait, value in profile.items():
            if value == "high":
                for (trait_level, trait_name), jobs in trait_to_jobs.items():
                    if trait_name.lower() in trait.lower() and trait_level == "high":
                        queries.extend(jobs)

        # Secondary queries from keywords
        keywords = profile.get("keywords", [])
        if keywords:
            queries.extend(keywords[:3])

        # Tertiary queries based on environment preference
        if profile.get("environment_preference") == "fast_paced":
            queries.extend(["Startup", "Consultant", "Operations Manager"])
        elif profile.get("environment_preference") == "flexible":
            queries.extend(["Remote Worker", "Contractor", "Consultant"])

        # Remove duplicates and limit
        queries = list(set(queries))[:5]

        return queries if queries else ["Software Engineer", "Project Manager"]

    def _deduplicate_jobs(self, jobs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate job postings"""
        seen = set()
        unique = []

        for job in jobs:
            job_id = (job.get("title"), job.get("company"), job.get("location"))
            if job_id not in seen:
                seen.add(job_id)
                unique.append(job)

        return unique

    def get_search_history(self) -> List[Dict[str, Any]]:
        """Return search history"""
        return self.search_history.copy()

    def reset(self) -> None:
        """Reset scout for new session"""
        self.search_history = []
        logger.info("ScoutAgent reset")
