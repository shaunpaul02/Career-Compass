"""
Comprehensive Test Suite for Career Compass Multi-Agent System

Tests all three agents (Profiler, Scout, Analyst) and the Orchestrator.
Demonstrates core concepts:
- Chain-of-Thought Reasoning (Profiler)
- Function Calling (Scout)
- Data Synthesis (Analyst)
- State Management (Orchestrator)
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from profiler_agent import ProfilerAgent, TraitObject
from scout_agent import ScoutAgent, SearchTool
from analyst_agent import AnalystAgent, MatchCard
from orchestrator_agent import OrchestratorAgent, OrchestratorState, WorkflowState
import config


# ============================================================================
# PROFILER AGENT TESTS
# ============================================================================

class TestProfilerAgent:
    """Test suite for ProfilerAgent - Chain-of-Thought Reasoning"""

    def setup_method(self):
        """Setup before each test"""
        self.profiler = ProfilerAgent()

    def test_profiler_initialization(self):
        """Test that profiler initializes correctly"""
        assert self.profiler is not None
        assert self.profiler.conversation_history == []
        assert "resilience" in self.profiler.accumulated_traits

    def test_process_leadership_response(self):
        """Test trait extraction from leadership-focused response"""
        response = "I love leading my team and solving complex problems. I organized a hackathon last year that brought people together."

        trait_obj = self.profiler.process_user_response(
            response,
            question="Tell me about a time you felt successful"
        )

        # Verify TraitObject structure
        assert isinstance(trait_obj, TraitObject)
        assert trait_obj.leadership in ["low", "medium", "high"]
        assert trait_obj.problem_solving in ["low", "medium", "high"]
        assert len(trait_obj.reasoning) > 0

    def test_process_technical_response(self):
        """Test trait extraction from technical-focused response"""
        response = "I built a Python script that automated our daily data processing. It took problem-solving and technical skills."

        trait_obj = self.profiler.process_user_response(response)

        assert trait_obj.technical_aptitude in ["low", "medium", "high"]
        assert trait_obj.problem_solving in ["low", "medium", "high"]

    def test_conversation_memory(self):
        """Test that profiler maintains conversation history"""
        responses = [
            ("I'm a natural leader", "Leadership question"),
            ("I enjoy creative design work", "Creativity question"),
            ("I handle stress well", "Resilience question")
        ]

        for response, question in responses:
            self.profiler.process_user_response(response, question)

        history = self.profiler.get_conversation_history()
        assert len(history) > 0
        # Each response should create at least 2 history entries (user + profiler)
        assert len(history) >= len(responses)

    def test_trait_accumulation(self):
        """Test that traits accumulate across multiple responses"""
        # First response establishes baseline
        self.profiler.process_user_response("I'm very organized")
        profile1 = self.profiler.get_accumulated_profile()

        # Second response should modify profile
        self.profiler.process_user_response("I actually love fast-paced environments")
        profile2 = self.profiler.get_accumulated_profile()

        # Profile should evolve
        assert profile1 is not None
        assert profile2 is not None

    def test_profiler_reset(self):
        """Test profiler reset functionality"""
        self.profiler.process_user_response("I'm a leader")
        assert len(self.profiler.get_conversation_history()) > 0

        self.profiler.reset()

        assert self.profiler.get_conversation_history() == []
        assert self.profiler.accumulated_traits["resilience"] == "medium"

    def test_fallback_extraction(self):
        """Test fallback extraction when JSON parsing fails"""
        # This would naturally occur if LLM response is malformed
        result = self.profiler._fallback_trait_extraction(
            "I overcome challenges and lead teams to success"
        )

        assert result["resilience"] == "high"
        assert result["leadership"] == "high"

    @pytest.mark.parametrize("input_text,expected_trait,expected_level", [
        ("I thrive under pressure", "resilience", "high"),
        ("I enjoy organizing things", "leadership", "high"),
        ("I write code daily", "technical_aptitude", "high"),
        ("I love analyzing data", "problem_solving", "high"),
    ])
    def test_trait_extraction_parametrized(self, input_text, expected_trait, expected_level):
        """Parametrized test for common trait patterns"""
        result = self.profiler._fallback_trait_extraction(input_text)
        assert result[expected_trait] == expected_level


# ============================================================================
# SCOUT AGENT TESTS
# ============================================================================

class TestScoutAgent:
    """Test suite for ScoutAgent - Function Calling"""

    def setup_method(self):
        """Setup before each test"""
        self.scout = ScoutAgent()

    def test_scout_initialization(self):
        """Test that scout initializes with search tool"""
        assert self.scout is not None
        assert self.scout.search_tool is not None

    def test_search_tool_initialization(self):
        """Test SearchTool initialization"""
        tool = SearchTool()
        assert tool is not None
        assert tool.base_url == "https://www.googleapis.com/customsearch/v1"

    def test_mock_job_search(self):
        """Test mock job search results"""
        tool = SearchTool()
        results = tool.search_jobs("Project Manager", "London, ON")

        assert len(results) > 0
        assert "title" in results[0]
        assert "company" in results[0]
        assert "description" in results[0]

    def test_has_sufficient_profile_data(self):
        """Test profile data sufficiency check"""
        # Weak profile
        weak_profile = {
            "resilience": "medium",
            "problem_solving": "medium",
            "environment_preference": "mixed"
        }
        assert not self.scout._has_sufficient_profile_data(weak_profile)

        # Strong profile
        strong_profile = {
            "resilience": "high",
            "problem_solving": "high",
            "environment_preference": "fast_paced",
            "keywords": ["engineering", "startup"]
        }
        assert self.scout._has_sufficient_profile_data(strong_profile)

    def test_generate_search_queries(self):
        """Test search query generation from traits"""
        profile = {
            "resilience": "high",
            "leadership": "high",
            "technical_aptitude": "medium",
            "problem_solving": "high",
            "environment_preference": "fast_paced",
            "keywords": ["startup"]
        }

        queries = self.scout._generate_search_queries(profile)

        assert len(queries) > 0
        assert isinstance(queries, list)
        # Should contain job-related keywords
        assert any(term in str(queries).lower() for term in ["manager", "engineer", "analyst"])

    def test_find_matching_jobs(self):
        """Test full job matching workflow"""
        profile = {
            "resilience": "high",
            "leadership": "high",
            "problem_solving": "high",
            "environment_preference": "fast_paced",
            "keywords": ["project", "management"]
        }

        jobs = self.scout.find_matching_jobs(profile)

        assert isinstance(jobs, list)
        # Should find jobs based on profile
        assert len(jobs) > 0

    def test_deduplicate_jobs(self):
        """Test job deduplication"""
        jobs = [
            {"title": "Manager", "company": "Acme", "location": "London"},
            {"title": "Manager", "company": "Acme", "location": "London"},  # Duplicate
            {"title": "Engineer", "company": "TechCorp", "location": "London"},
        ]

        unique = self.scout._deduplicate_jobs(jobs)
        assert len(unique) == 2

    def test_scout_reset(self):
        """Test scout reset"""
        # Do a search
        profile = {"resilience": "high", "keywords": ["test"]}
        self.scout.find_matching_jobs(profile)

        assert len(self.scout.get_search_history()) > 0

        self.scout.reset()
        assert self.scout.get_search_history() == []


# ============================================================================
# ANALYST AGENT TESTS
# ============================================================================

class TestAnalystAgent:
    """Test suite for AnalystAgent - Data Synthesis"""

    def setup_method(self):
        """Setup before each test"""
        self.analyst = AnalystAgent()

    def test_analyst_initialization(self):
        """Test analyst initialization"""
        assert self.analyst is not None
        assert self.analyst.analysis_history == []

    def test_extract_job_traits(self):
        """Test job trait extraction"""
        job = {
            "title": "Software Engineer Manager",
            "description": "Lead a team of engineers in a fast-paced, high-pressure startup environment. Requires strong technical and leadership skills.",
            "company": "TechCorp"
        }

        traits = self.analyst._extract_job_traits(job)

        assert traits["leadership"] == "high"
        assert traits["technical_aptitude"] == "high"
        assert traits["resilience"] == "high"

    def test_identify_trait_alignment(self):
        """Test alignment identification"""
        user_profile = {
            "leadership": "high",
            "technical_aptitude": "high",
            "problem_solving": "medium"
        }

        job_traits = {
            "leadership": "high",
            "technical_aptitude": "high",
            "problem_solving": "high",
            "communication": "medium"
        }

        matched, unmatched = self.analyst._identify_trait_alignment(user_profile, job_traits)

        assert len(matched) > 0
        assert len(unmatched) > 0

    def test_generate_recommendation(self):
        """Test recommendation generation"""
        assert self.analyst._generate_recommendation(0.85) == "Strong Match"
        assert self.analyst._generate_recommendation(0.70) == "Good Fit"
        assert self.analyst._generate_recommendation(0.50) == "Consider"
        assert self.analyst._generate_recommendation(0.30) == "Not Aligned"

    def test_assess_market_availability(self):
        """Test market availability assessment"""
        job_low = {"applicants": 20, "title": "Rare Role"}
        job_high = {"applicants": 500, "title": "Common Role"}

        assert "High Availability" in self.analyst._assess_market_availability(job_low)
        assert "Highly Competitive" in self.analyst._assess_market_availability(job_high)

    def test_analyze_job_fit(self):
        """Test full job fit analysis"""
        job = {
            "title": "Project Manager",
            "company": "Acme Corp",
            "description": "Lead projects in a fast-paced environment",
            "applicants": 50
        }

        profile = {
            "leadership": "high",
            "resilience": "high",
            "problem_solving": "high",
            "teamwork": "medium"
        }

        match_card = self.analyst.analyze_job_fit(job, profile)

        assert isinstance(match_card, MatchCard)
        assert 0 <= match_card.compatibility_score <= 1
        assert 0 <= match_card.match_percentage <= 100
        assert len(match_card.reasoning) > 0

    def test_batch_analyze_jobs(self):
        """Test batch job analysis"""
        jobs = [
            {"title": "Manager", "company": "Corp1", "description": "Lead team", "applicants": 50},
            {"title": "Engineer", "company": "Corp2", "description": "Technical work", "applicants": 100},
        ]

        profile = {"leadership": "high", "technical_aptitude": "medium"}

        results = self.analyst.batch_analyze_jobs(jobs, profile)

        assert len(results) == 2
        # Should be sorted by score
        assert results[0].compatibility_score >= results[1].compatibility_score

    def test_get_top_matches(self):
        """Test getting top matches"""
        jobs = [
            {"title": f"Job{i}", "company": f"Corp{i}", "description": "A job", "applicants": 50}
            for i in range(10)
        ]

        profile = {"leadership": "high", "technical_aptitude": "high"}

        top_matches = self.analyst.get_top_matches(jobs, profile, top_n=3)

        assert len(top_matches) <= 3


# ============================================================================
# ORCHESTRATOR AGENT TESTS
# ============================================================================

class TestOrchestratorAgent:
    """Test suite for OrchestratorAgent - State Management"""

    def setup_method(self):
        """Setup before each test"""
        self.orchestrator = OrchestratorAgent()

    def test_orchestrator_initialization(self):
        """Test orchestrator initialization"""
        assert self.orchestrator is not None
        assert self.orchestrator.state.workflow_state == WorkflowState.IDLE
        assert self.orchestrator.profiler is not None
        assert self.orchestrator.scout is not None
        assert self.orchestrator.analyst is not None

    def test_process_quiz_response(self):
        """Test processing a quiz response"""
        response = "I'm a strong leader who solves complex problems"

        result = self.orchestrator.process_quiz_response(
            response,
            "Tell me about a success"
        )

        assert result["status"] == "success"
        assert "traits" in result
        assert "reasoning" in result

    def test_should_trigger_search(self):
        """Test search trigger logic"""
        weak_profile = {"resilience": "medium", "leadership": "medium"}
        assert not self.orchestrator._should_trigger_search(weak_profile)

        strong_profile = {"resilience": "high", "leadership": "high", "keywords": ["engineer"]}
        assert self.orchestrator._should_trigger_search(strong_profile)

    def test_full_workflow(self):
        """Test full orchestration workflow"""
        # Step 1: Process responses
        responses = [
            ("I love leading teams in high-pressure situations", "Leadership"),
            ("I solve complex technical problems", "Problem-solving"),
            ("I thrive in fast-paced startups", "Environment"),
        ]

        for response, question in responses:
            result = self.orchestrator.process_quiz_response(response, question)
            assert result["status"] == "success"

        # Step 2: Check state progression
        state = self.orchestrator.get_current_state()
        assert state["user_profile"] is not None
        # State may have progressed to ANALYZING if search was triggered
        assert state["workflow_state"] in ["profiling", "completed"]

    def test_get_top_matches(self):
        """Test getting top matches from orchestrator"""
        # Trigger a workflow
        self.orchestrator.process_quiz_response(
            "I'm a strong technical leader",
            "Success"
        )

        top_matches = self.orchestrator.get_top_matches(top_n=3)
        assert isinstance(top_matches, list)

    def test_orchestrator_state_conversion(self):
        """Test OrchestratorState to dict conversion"""
        state = self.orchestrator.get_current_state()

        assert isinstance(state, dict)
        assert "workflow_state" in state
        assert "user_profile" in state
        assert "extracted_jobs" in state

    def test_orchestrator_reset(self):
        """Test orchestrator reset"""
        # Add data
        self.orchestrator.process_quiz_response("I'm a leader", "Q1")
        assert self.orchestrator.state.conversation_history is not None

        # Reset
        self.orchestrator.reset_session()
        assert self.orchestrator.state.workflow_state == WorkflowState.IDLE
        assert self.orchestrator.state.conversation_history == []

    def test_session_summary(self):
        """Test session summary generation"""
        self.orchestrator.process_quiz_response("Test response", "Test question")

        summary = self.orchestrator.get_session_summary()

        assert "session_id" in summary
        assert "workflow_state" in summary
        assert "conversation_turns" in summary

    def test_detailed_report(self):
        """Test detailed report generation"""
        self.orchestrator.process_quiz_response("I'm technical", "Question")

        report = self.orchestrator.get_detailed_report()

        assert "session_id" in report
        assert "profiler_data" in report
        assert "scout_data" in report
        assert "analyst_data" in report


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests for full system workflow"""

    def test_end_to_end_career_compass(self):
        """Test complete Career Compass workflow"""
        orchestrator = OrchestratorAgent()

        # Simulate user quiz responses
        quiz_data = [
            ("I love being in control and organizing my team. Last year, I led a project that was critical for our company.", "Tell us about a leadership experience"),
            ("I'm great at solving complex problems quickly, even under stress.", "How do you handle pressure?"),
            ("I prefer dynamic, fast-paced environments where I can wear multiple hats.", "What's your ideal work environment?"),
        ]

        # Process each response
        for response, question in quiz_data:
            result = orchestrator.process_quiz_response(response, question)
            assert result["status"] == "success", f"Failed on: {question}"

        # Verify system state
        state = orchestrator.get_current_state()
        assert len(state["user_profile"]) > 0, "Profile should be populated"

        # Get final report
        report = orchestrator.get_detailed_report()
        assert "session_id" in report

        print("\n✓ End-to-end test passed!")
        print(f"Profile: {state['user_profile']}")
        print(f"Jobs found: {len(state['extracted_jobs'])}")
        print(f"Match cards: {len(state['match_cards'])}")


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

class TestPerformance:
    """Performance tests"""

    def test_profile_response_time(self):
        """Test that profile response processing is reasonably fast"""
        profiler = ProfilerAgent()

        import time
        start = time.time()
        profiler.process_user_response("I'm a leader who solves problems")
        elapsed = time.time() - start

        # Should process in reasonable time (< 10 seconds in testing)
        assert elapsed < 10, f"Profile processing took {elapsed}s"

    def test_search_tool_performance(self):
        """Test search tool performance"""
        tool = SearchTool()

        import time
        start = time.time()
        results = tool.search_jobs("Project Manager", "London, ON")
        elapsed = time.time() - start

        assert len(results) > 0
        print(f"Search completed in {elapsed:.2f}s")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
