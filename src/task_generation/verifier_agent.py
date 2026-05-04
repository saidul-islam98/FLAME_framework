"""Source file for the verifier agent used in task verification."""

import logging
from typing import Any, Dict, Tuple, Union, cast

from src.task_generation.json_response_utils import parse_json_like, stringify_payload
from src.task_generation.prompts import (
    MCQ_INTEGRITY_OUTPUT_FORMAT,
    SYSTEM_MCQ_INTEGRITY_CHECK_AND_REVISE_PROMPT,
    SYSTEM_TASK_VERIFICATION_PROMPT,
    USER_MCQ_INTEGRITY_CHECK_AND_REVISE_PROMPT,
    USER_TASK_VERIFICATION_PROMPT,
)
from src.utils.model_client_utils import (
    ModelCallMode,
    TextWithUsage,
    async_call_model,
)


logger = logging.getLogger(__name__)


class VerifierAgent:
    """Verifier agent for task verification."""

    def __init__(self, name: str, model_client: Any, **kwargs: Any) -> None:
        self.name = name
        self.model_client = model_client

    async def verify_task(
        self,
        candidate_output: Union[Dict[str, Any], str],
        blooms_level: str,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Verify the candidate task output against the provided blueprint and context.

        Args:
            candidate_output: The candidate task output to be verified.

        Returns
        -------
            A dictionary containing the verification results.
        """
        candidate_str = stringify_payload(candidate_output)

        user_prompt = USER_TASK_VERIFICATION_PROMPT.format(
            candidate_output=candidate_str,
            blooms_level=(blooms_level or "").strip(),
        )
        task = SYSTEM_TASK_VERIFICATION_PROMPT + "\n\n" + user_prompt
        result = await self._call_text_prompt(task)
        return self._extract_verification_report(str(result["content"])), dict(
            result["usage"]
        )

    async def check_and_revise_mcq_option(
        self,
        candidate_question: str,
        blooms_level: str,
        solution_trace: Dict[str, Any],
        solution_full: Dict[str, Any],
    ) -> Tuple[Union[Dict[str, Any], str], str, Dict[str, Any]]:
        """
        Check the correctness of a candidate question.

        Args:
            candidate_question: The candidate problem to process.
            blooms_level: Target Bloom's level for the candidate question.
            solution_trace: The step-by-step solution trace for the candidate question.
            solution_full: The full solution for the candidate question.

        Returns
        -------
            A tuple containing the processed problems and the full prompt used.
        """
        task = USER_MCQ_INTEGRITY_CHECK_AND_REVISE_PROMPT.format(
            candidate_question=candidate_question,
            blooms_level=(blooms_level or "").strip(),
            solution_trace=stringify_payload(solution_trace),
            solution_full=stringify_payload(solution_full),
        )
        task = (
            SYSTEM_MCQ_INTEGRITY_CHECK_AND_REVISE_PROMPT
            + "\n\n"
            + task
            + "\n\n"
            + MCQ_INTEGRITY_OUTPUT_FORMAT
        )
        result = await self._call_text_prompt(task)
        return (
            self._extract_mcq_payload(str(result["content"])),
            task,
            dict(result["usage"]),
        )

    async def _call_text_prompt(self, task: str) -> TextWithUsage:
        """Call the model in plain-text mode for a fully assembled prompt."""
        response = await async_call_model(
            self.model_client,
            user_prompt=task,
            mode=ModelCallMode.TEXT,
            return_usage=True,
        )
        assert isinstance(response, dict)
        return cast(TextWithUsage, response)

    def _extract_mcq_payload(self, content: str) -> Union[Dict[str, Any], str]:
        """
        Extract payload for Step-3 (MCQ repair path).

        No verifier-report fallback here; return {} on parse failure.
        """
        obj = parse_json_like(content)
        if isinstance(obj, dict):
            return obj
        if obj is None:
            logger.warning("Verifier MCQ payload parse failed; returning empty dict.")
            return {}
        return {"result": obj}

    def _extract_verification_report(self, content: str) -> Dict[str, Any]:
        """
        Extract payload for Step-7 (verification report path).

        Uses fail-report fallback to keep verification loop deterministic.
        """
        obj = parse_json_like(content)
        if isinstance(obj, dict):
            return obj
        return self._fallback_report("json parse failed")

    def _fallback_report(self, msg: str) -> Dict[str, Any]:
        """Structured fail report fallback for verification stage only."""
        return {
            "overall_verdict": "Fail",
            "json_format_valid": "No",
            "mcq_integrity": "No",
            "blooms_alignment": "No",
            "constraint_compliance": "No",
            "explanation": f"System Error: {msg}",
            "question_evaluation": {
                "distractors_plausible": "No",
                "main_issues": [f"parse_error: {msg}"],
                "fix": "Return a single valid JSON object matching the verification schema.",
            },
        }
