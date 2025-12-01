"""
Orchestrator Agent: Coordinates the Profiler, Scout, and Analyst agents.
Manages the workflow and state between agents.
Demonstrates LangGraph state management concepts.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum
from utils import get_logger, timestamp
from profiler_agent import ProfilerAgent
from scout_agent import ScoutAgent
from analyst_agent import AnalystAgent, MatchCard
import json

logger = get_logger(__name__)


class WorkflowState(Enum):
    """States in the Career Compass workflow"""
    IDLE = "idle"
    PROFILING = "profiling"
    SEARCHING = "searching"
    ANALYZING = "analyzing"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class OrchestratorState:
    """Encapsulates the state flowing through the system"""
    workflow_state: WorkflowState
    user_profile: Dict[str, Any]
    extracted_jobs: List[Dict[str, Any]]
    match_cards: List[Dict[str, Any]]
    conversation_history: List[Dict[str, Any]]
    errors: List[str]
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "workflow_state": self.workflow_state.value,
            "user_profile": self.user_profile,
            "extracted_jobs": self.extracted_jobs,
            "match_cards": self.match_cards,
            "conversation_history": self.conversation_history,
            "errors": self.errors,
            "timestamp": self.timestamp
        }


class OrchestratorAgent:
    """
    Coordinates the three specialized agents.

    Demonstrates:
    - State Management: Using OrchestratorState to track workflow
    - Agent Orchestration: Routing between Profiler, Scout, Analyst
    - Error Handling: Managing failures across agents
    """

    def __init__(self):
        self.profiler = ProfilerAgent()
        self.scout = ScoutAgent()
        self.analyst = AnalystAgent()
        self.state = OrchestratorState(
            workflow_state=WorkflowState.IDLE,
            user_profile={},
            extracted_jobs=[],
            match_cards=[],
            conversation_history=[],
            errors=[],
            timestamp=timestamp()
        )
        self.session_id = f"session_{timestamp().replace(':', '-')}"
        logger.info(f"OrchestratorAgent initialized with session: {self.session_id}")

    def process_quiz_response(self, user_input: str, question: str = "") -> Dict[str, Any]:
        """
        Process a user's quiz response through the Profiler Agent.

        Args:
            user_input: User's natural language response
            question: The quiz question

        Returns:
            State update reflecting profiler output
        """
        logger.info(">>> Orchestrator: Processing quiz response")
        self.state.workflow_state = WorkflowState.PROFILING

        try:
            # Step 1: Profile the user
            trait_object = self.profiler.process_user_response(user_input, question)
            logger.info(f"Profiler output: {trait_object.dict()}")

            # Step 2: Update accumulated profile
            self.state.user_profile = self.profiler.get_accumulated_profile()

            # Step 3: Check if profile is sufficient for searching
            should_search = self._should_trigger_search(self.state.user_profile)

            if should_search:
                logger.info("Profile data sufficient, triggering Scout Agent")
                self._execute_search_workflow()

            self.state.conversation_history = self.profiler.get_conversation_history()

            return {
                "status": "success",
                "workflow_state": self.state.workflow_state.value,
                "traits": trait_object.dict(),
                "should_search": should_search,
                "reasoning": trait_object.reasoning
            }

        except Exception as e:
            logger.error(f"Error in process_quiz_response: {str(e)}")
            self.state.workflow_state = WorkflowState.ERROR
            self.state.errors.append(f"Quiz processing error: {str(e)}")
            return {
                "status": "error",
                "message": str(e),
                "workflow_state": self.state.workflow_state.value
            }

    def _should_trigger_search(self, profile: Dict[str, Any]) -> bool:
        """
        Determine if profile has enough data to trigger job search.
        Demonstrates autonomous agent decision-making.
        """
        # Check for meaningful trait deductions (not all "medium")
        significant_traits = [
            profile.get("resilience"),
            profile.get("leadership"),
            profile.get("technical_aptitude")
        ]

        non_medium = sum(1 for trait in significant_traits if trait != "medium")
        has_keywords = len(profile.get("keywords", [])) > 0

        return non_medium >= 2 or has_keywords

    def _execute_search_workflow(self) -> None:
        """
        Execute the full search workflow: Scout -> Analyst
        """
        logger.info(">>> Orchestrator: Executing search workflow")
        self.state.workflow_state = WorkflowState.SEARCHING

        try:
            # Step 1: Scout finds jobs
            jobs = self.scout.find_matching_jobs(self.state.user_profile)
            self.state.extracted_jobs = jobs

            logger.info(f"Scout found {len(jobs)} jobs")

            if jobs:
                # Step 2: Analyst ranks jobs
                self.state.workflow_state = WorkflowState.ANALYZING
                match_cards = self.analyst.batch_analyze_jobs(jobs, self.state.user_profile)

                # Convert MatchCard objects to dicts for serialization
                self.state.match_cards = [mc.dict() for mc in match_cards]
                logger.info(f"Analyst ranked {len(match_cards)} jobs")

            self.state.workflow_state = WorkflowState.COMPLETED

        except Exception as e:
            logger.error(f"Error in search workflow: {str(e)}")
            self.state.workflow_state = WorkflowState.ERROR
            self.state.errors.append(f"Search workflow error: {str(e)}")

    def get_current_state(self) -> Dict[str, Any]:
        """Get the current orchestrator state"""
        return self.state.to_dict()

    def get_top_matches(self, top_n: int = 5) -> List[Dict[str, Any]]:
        """Get top N job matches from current state"""
        return self.state.match_cards[:top_n]

    def reset_session(self) -> None:
        """Reset all agents and start new session"""
        logger.info("Resetting session")
        self.profiler.reset()
        self.scout.reset()
        self.analyst.reset()
        self.state = OrchestratorState(
            workflow_state=WorkflowState.IDLE,
            user_profile={},
            extracted_jobs=[],
            match_cards=[],
            conversation_history=[],
            errors=[],
            timestamp=timestamp()
        )
        self.session_id = f"session_{timestamp().replace(':', '-')}"

    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of the session"""
        return {
            "session_id": self.session_id,
            "workflow_state": self.state.workflow_state.value,
            "profile_traits": self.state.user_profile,
            "jobs_found": len(self.state.extracted_jobs),
            "top_matches": len(self.state.match_cards),
            "conversation_turns": len(self.state.conversation_history),
            "errors": self.state.errors
        }

    def get_detailed_report(self) -> Dict[str, Any]:
        """Get detailed report of the entire session"""
        return {
            "session_id": self.session_id,
            "timestamp": self.state.timestamp,
            "workflow_state": self.state.workflow_state.value,

            "profiler_data": {
                "accumulated_profile": self.state.user_profile,
                "conversation_history": self.profiler.get_conversation_history()
            },

            "scout_data": {
                "jobs_found": len(self.state.extracted_jobs),
                "search_history": self.scout.get_search_history()
            },

            "analyst_data": {
                "match_cards": self.state.match_cards,
                "analysis_history": self.analyst.get_analysis_history()
            },

            "errors": self.state.errors
        }
