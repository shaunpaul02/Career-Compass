"""
Profiler Agent: Responsible for parsing user responses and extracting traits.
Uses Chain-of-Thought reasoning to decompose user answers into structured trait objects.
"""

from typing import Dict, Any, List
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from utils import get_logger, timestamp
import json
import config

logger = get_logger(__name__)

class TraitObject(BaseModel):
    """Structured representation of extracted user traits"""
    resilience: str = Field(description="How well user handles stress/adversity (low/medium/high)")
    leadership: str = Field(description="Leadership capabilities (low/medium/high)")
    technical_aptitude: str = Field(description="Technical skill level (low/medium/high)")
    problem_solving: str = Field(description="Problem-solving ability (low/medium/high)")
    teamwork: str = Field(description="Collaboration skills (low/medium/high)")
    environment_preference: str = Field(description="Preferred work environment (fast_paced/structured/flexible)")
    communication: str = Field(description="Communication skills (low/medium/high)")
    creativity: str = Field(description="Creative thinking (low/medium/high)")
    work_style: str = Field(description="Work style preference (independent/collaborative/mixed)")
    extracted_keywords: List[str] = Field(description="Key phrases from user input")
    reasoning: str = Field(description="Chain-of-Thought explanation for deductions")


class ProfilerAgent:
    """
    The Profiler Agent processes user quiz responses and extracts traits.

    Demonstrates:
    - Chain-of-Thought Reasoning: Breaking down complex user inputs
    - Conversational Memory: Maintaining state across multiple questions
    - Structured Output: Returning JSON-compatible trait objects
    """

    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model=config.MODEL_NAME,
            temperature=config.TEMPERATURE,
            google_api_key=config.GOOGLE_API_KEY
        )
        self.conversation_history: List[Dict[str, str]] = []
        self.accumulated_traits: Dict[str, Any] = {
            "resilience": "medium",
            "leadership": "medium",
            "technical_aptitude": "medium",
            "problem_solving": "medium",
            "teamwork": "medium",
            "environment_preference": "mixed",
            "communication": "medium",
            "creativity": "medium",
            "work_style": "mixed",
            "keywords": [],
            "job_titles": []
        }
        logger.info("ProfilerAgent initialized")

    def _create_reasoning_prompt(self, user_input: str) -> str:
        """Create a Chain-of-Thought prompt for trait extraction"""

        system_context = f"""You are a career psychologist analyzing a user's response to understand their professional traits.

Current Accumulated Traits (from previous responses):
{json.dumps(self.accumulated_traits, indent=2)}

User's Response: "{user_input}"

Your task:
1. ANALYZE the response deeply. What does this tell us about the user?
2. EXTRACT specific traits mentioned or implied (e.g., "I love solving puzzles" → high problem_solving)
3. INFER work environment preferences (e.g., "I thrive under pressure" → fast_paced)
4. IDENTIFY keywords that could be job titles or industries
5. EXPLAIN your reasoning in a natural way

Return ONLY valid JSON matching this structure (no markdown, no explanation outside JSON):
{{
    "resilience": "low|medium|high",
    "leadership": "low|medium|high",
    "technical_aptitude": "low|medium|high",
    "problem_solving": "low|medium|high",
    "teamwork": "low|medium|high",
    "environment_preference": "fast_paced|structured|flexible|mixed",
    "communication": "low|medium|high",
    "creativity": "low|medium|high",
    "work_style": "independent|collaborative|mixed",
    "extracted_keywords": ["keyword1", "keyword2", ...],
    "reasoning": "Your Chain-of-Thought explanation here..."
}}"""

        return system_context

    def process_user_response(self, user_input: str, question: str = "") -> TraitObject:
        """
        Process a user's response to a quiz question.

        Args:
            user_input: The user's natural language response
            question: The question that was asked (for context)

        Returns:
            TraitObject: Structured traits extracted from the response
        """
        logger.info(f"Processing user response: {user_input[:100]}...")

        # Store in conversation history
        self.conversation_history.append({
            "timestamp": timestamp(),
            "question": question,
            "response": user_input,
            "role": "user"
        })

        prompt = self._create_reasoning_prompt(user_input)

        try:
            # Call LLM with reasoning prompt
            response = self.llm.invoke(prompt)
            response_text = response.content.strip()

            # Try to parse JSON from response
            try:
                # Sometimes LLM wraps JSON in markdown, so clean it
                if "```json" in response_text:
                    response_text = response_text.split("```json")[1].split("```")[0]
                elif "```" in response_text:
                    response_text = response_text.split("```")[1].split("```")[0]

                traits_dict = json.loads(response_text)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON response: {response_text[:200]}")
                traits_dict = self._fallback_trait_extraction(user_input)

            # Create TraitObject
            trait_obj = TraitObject(**traits_dict)

            # Update accumulated traits (merge with previous)
            self._merge_traits(trait_obj)

            # Store in history
            self.conversation_history.append({
                "timestamp": timestamp(),
                "traits": trait_obj.dict(),
                "role": "profiler"
            })

            logger.info(f"Extracted traits: {trait_obj.dict()}")
            return trait_obj

        except Exception as e:
            logger.error(f"Error processing response: {str(e)}")
            # Return default trait object on error
            return self._create_default_traits()

    def _merge_traits(self, new_traits: TraitObject) -> None:
        """
        Merge new traits with accumulated traits, giving weight to more recent responses.
        """
        # Update scalar traits
        for field in ["resilience", "leadership", "technical_aptitude", "problem_solving",
                      "teamwork", "communication", "creativity"]:
            new_val = getattr(new_traits, field, "medium")
            if new_val != "medium":
                self.accumulated_traits[field] = new_val

        # Update categorical traits
        if new_traits.environment_preference != "mixed":
            self.accumulated_traits["environment_preference"] = new_traits.environment_preference

        if new_traits.work_style != "mixed":
            self.accumulated_traits["work_style"] = new_traits.work_style

        # Append keywords
        self.accumulated_traits["keywords"].extend(new_traits.extracted_keywords)
        self.accumulated_traits["keywords"] = list(set(self.accumulated_traits["keywords"]))  # Deduplicate

    def _fallback_trait_extraction(self, user_input: str) -> Dict[str, Any]:
        """
        Fallback extraction if JSON parsing fails.
        Uses simple keyword matching.
        """
        logger.warning("Using fallback trait extraction")

        input_lower = user_input.lower()
        traits = {
            "resilience": "high" if any(word in input_lower for word in ["overcome", "challenge", "difficult", "pressure"]) else "medium",
            "leadership": "high" if any(word in input_lower for word in ["lead", "manage", "team", "organize"]) else "medium",
            "technical_aptitude": "high" if any(word in input_lower for word in ["code", "build", "develop", "technical"]) else "medium",
            "problem_solving": "high" if any(word in input_lower for word in ["solve", "analyze", "debug", "figure"]) else "medium",
            "teamwork": "high" if any(word in input_lower for word in ["team", "collaborate", "together", "group"]) else "medium",
            "environment_preference": "mixed",
            "communication": "high" if any(word in input_lower for word in ["present", "speak", "communicate", "explain"]) else "medium",
            "creativity": "high" if any(word in input_lower for word in ["creative", "design", "innovate", "idea"]) else "medium",
            "work_style": "mixed",
            "extracted_keywords": [],
            "reasoning": "Fallback extraction due to parsing error"
        }
        return traits

    def _create_default_traits(self) -> TraitObject:
        """Create a default trait object"""
        return TraitObject(
            resilience="medium",
            leadership="medium",
            technical_aptitude="medium",
            problem_solving="medium",
            teamwork="medium",
            environment_preference="mixed",
            communication="medium",
            creativity="medium",
            work_style="mixed",
            extracted_keywords=[],
            reasoning="Default traits - error in processing"
        )

    def get_accumulated_profile(self) -> Dict[str, Any]:
        """Return the accumulated profile from all responses"""
        return self.accumulated_traits.copy()

    def get_conversation_history(self) -> List[Dict]:
        """Return the conversation history"""
        return self.conversation_history.copy()

    def reset(self) -> None:
        """Reset the profiler for a new session"""
        self.conversation_history = []
        self.accumulated_traits = {
            "resilience": "medium",
            "leadership": "medium",
            "technical_aptitude": "medium",
            "problem_solving": "medium",
            "teamwork": "medium",
            "environment_preference": "mixed",
            "communication": "medium",
            "creativity": "medium",
            "work_style": "mixed",
            "keywords": [],
            "job_titles": []
        }
        logger.info("ProfilerAgent reset")
