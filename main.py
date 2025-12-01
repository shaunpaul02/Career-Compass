"""
Career Compass: Main Application Entry Point

This is the central orchestration point that demonstrates the multi-agent system
for intelligent career matching. It can be run as:
1. CLI interface for testing
2. API backend for React frontend
3. Batch processor for testing multiple profiles
"""

from orchestrator_agent import OrchestratorAgent
from utils import get_logger
import json
import sys
from typing import List, Dict, Any

logger = get_logger(__name__)


class CareerCompass:
    """Main application controller"""

    def __init__(self):
        self.orchestrator = OrchestratorAgent()
        logger.info("Career Compass initialized")

    def run_interactive_quiz(self) -> Dict[str, Any]:
        """
        Run interactive quiz mode where user answers questions.
        Returns final career recommendations.
        """
        print("\n" + "="*70)
        print("  CAREER COMPASS: Intelligent Career Matching System")
        print("="*70)
        print("\nWelcome! Let's discover your ideal career path.\n")

        # Define quiz questions
        questions = [
            {
                "question": "Tell us about a time you felt most successful or accomplished.",
                "hint": "(Focus on what skills or traits made you successful)"
            },
            {
                "question": "How do you typically react when faced with high-pressure situations?",
                "hint": "(Describe your approach and how you handle stress)"
            },
            {
                "question": "Describe your ideal work environment and team dynamic.",
                "hint": "(Fast-paced? Structured? Independent? Collaborative?)"
            },
            {
                "question": "What types of problems or challenges energize you most?",
                "hint": "(Technical? People-related? Strategic? Creative?)"
            },
            {
                "question": "Tell us about your career aspirations and what success looks like to you.",
                "hint": "(Growth trajectory, impact, compensation, work-life balance, etc.)"
            }
        ]

        # Process each question
        for i, q in enumerate(questions, 1):
            print(f"\n[Question {i}/{len(questions)}]")
            print(f"{q['question']}")
            print(f"Hint: {q['hint']}")
            print("-" * 70)

            user_response = input("Your response: ").strip()

            if not user_response:
                print("(Skipping empty response)")
                continue

            print("\nAnalyzing your response...")
            result = self.orchestrator.process_quiz_response(
                user_response,
                q["question"]
            )

            if result["status"] == "success":
                print(f"✓ Traits extracted: {result['reasoning'][:100]}...")

                if result.get("should_search"):
                    print("✓ Searching for matching jobs...")
            else:
                print(f"✗ Error: {result.get('message', 'Unknown error')}")

        print("\n" + "="*70)
        print("Analysis complete! Generating recommendations...\n")

        return self._generate_recommendations()

    def _generate_recommendations(self) -> Dict[str, Any]:
        """Generate and display career recommendations"""
        top_matches = self.orchestrator.get_top_matches(top_n=5)
        session_summary = self.orchestrator.get_session_summary()

        print("\n" + "="*70)
        print("  YOUR CAREER PROFILE")
        print("="*70)

        # Display profile summary
        profile = session_summary.get("profile_traits", {})
        print("\nKey Traits:")
        for trait, value in profile.items():
            if trait not in ["keywords", "job_titles"]:
                print(f"  • {trait.replace('_', ' ').title()}: {value}")

        if profile.get("keywords"):
            print(f"\nCareer Keywords: {', '.join(profile['keywords'][:5])}")

        print("\n" + "="*70)
        print("  TOP JOB MATCHES")
        print("="*70 + "\n")

        # Display job matches
        if not top_matches:
            print("No job matches found. Try providing more detailed responses.")
        else:
            for i, job in enumerate(top_matches, 1):
                print(f"[{i}] {job['job_title']} at {job['company']}")
                print(f"    Match: {job['match_percentage']}% - {job['recommendation']}")
                print(f"    Why: {job['reasoning'][:150]}...")
                print(f"    Market: {job['market_availability']}")
                print(f"    Strengths: {', '.join(job['key_strengths'][:2])}")
                print(f"    Growth Areas: {', '.join(job['growth_areas'][:2])}")
                print()

        return {
            "status": "success",
            "summary": session_summary,
            "recommendations": top_matches
        }

    def run_batch_test(self, test_profiles: List[Dict[str, str]]) -> None:
        """
        Run batch testing with multiple pre-defined profiles.
        Useful for demonstrating system capabilities.
        """
        logger.info(f"Running batch test with {len(test_profiles)} profiles")

        results = []

        for profile_idx, profile in enumerate(test_profiles, 1):
            print(f"\n>>> Processing Profile {profile_idx}/{len(test_profiles)}: {profile.get('name', 'Unknown')}")

            # Reset for new profile
            self.orchestrator.reset_session()

            # Process responses
            responses = profile.get("responses", [])
            for response in responses:
                self.orchestrator.process_quiz_response(response, "")

            # Collect results
            result = {
                "profile_name": profile.get("name"),
                "session_id": self.orchestrator.session_id,
                "summary": self.orchestrator.get_session_summary(),
                "top_matches": self.orchestrator.get_top_matches(top_n=3)
            }
            results.append(result)

            print(f"✓ Processed {len(responses)} responses")
            print(f"✓ Found {len(result['top_matches'])} top matches")

        # Save results
        self._save_batch_results(results)

    def _save_batch_results(self, results: List[Dict]) -> None:
        """Save batch test results to file"""
        output_file = "batch_test_results.json"

        # Convert to JSON-serializable format
        json_results = []
        for result in results:
            json_results.append({
                "profile_name": result["profile_name"],
                "session_id": result["session_id"],
                "summary": result["summary"],
                "matches_count": len(result["top_matches"])
            })

        with open(output_file, 'w') as f:
            json.dump(json_results, f, indent=2)

        logger.info(f"Batch results saved to {output_file}")
        print(f"\n✓ Results saved to {output_file}")

    def get_api_status(self) -> Dict[str, Any]:
        """Get system status for API health check"""
        return {
            "status": "healthy",
            "session_id": self.orchestrator.session_id,
            "workflow_state": self.orchestrator.get_current_state()["workflow_state"]
        }


def demo_profiles() -> List[Dict[str, Any]]:
    """Return predefined demo profiles for batch testing"""
    return [
        {
            "name": "Tech Leader",
            "responses": [
                "I led a team of 5 engineers through a critical project. I love guiding people and solving architectural problems.",
                "I handle pressure by staying calm and breaking problems into manageable pieces.",
                "I thrive in dynamic startup environments where I can wear multiple hats.",
                "I love technical challenges but also enjoy mentoring junior developers.",
                "I want to grow into a CTO role at an innovative tech company."
            ]
        },
        {
            "name": "Emergency Responder",
            "responses": [
                "I'm at my best when helping people in crisis situations. Last year I responded to an emergency and coordinated the response.",
                "High pressure energizes me. I stay focused and make quick, critical decisions.",
                "I prefer fast-paced, high-stakes environments where every second counts.",
                "I love solving urgent problems and being part of a tight-knit team.",
                "I want a career where I can make a real difference in people's lives."
            ]
        },
        {
            "name": "Creative Designer",
            "responses": [
                "I designed a complete UI overhaul that increased user engagement by 40%. I love the creative process.",
                "I'm calm under deadline pressure. I use creative thinking to overcome obstacles.",
                "I prefer collaborative environments with regular feedback and iteration.",
                "I'm energized by visual design problems and user experience challenges.",
                "I aspire to lead a design team at a product-focused company."
            ]
        }
    ]


def main():
    """Main entry point"""

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "demo":
            # Run demo with predefined profiles
            app = CareerCompass()
            app.run_batch_test(demo_profiles())

        elif command == "test":
            # Run test suite
            import subprocess
            subprocess.run(["python", "-m", "pytest", "test_agents.py", "-v"])

        elif command == "api":
            # Run API server (would integrate with Flask/FastAPI)
            print("API mode not yet implemented. See main.py for integration.")

        else:
            print(f"Unknown command: {command}")
            print("Usage: python main.py [interactive|demo|test|api]")
    else:
        # Run interactive mode by default
        app = CareerCompass()
        app.run_interactive_quiz()


if __name__ == "__main__":
    main()
