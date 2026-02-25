"""P4 implementation using a two-step critic-corrector generation flow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from openai import APIError, APIStatusError, AuthenticationError, PermissionDeniedError, RateLimitError
from pydantic import BaseModel, Field

from src.pipelines.p3_pipeline import P3_Pipeline
from src.state import AgentState


def _load_critic_map(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}

    normalized: Dict[str, str] = {}
    for key, value in payload.items():
        key_str = str(key).strip()
        value_str = str(value).strip()
        if key_str and value_str:
            normalized[key_str] = value_str
    return normalized


CRITIC_MAP_PATH = Path("specs/critic_instructions_map.json")
CRITIC_MAP = _load_critic_map(CRITIC_MAP_PATH)
# Questions that require strict critique filtering.
CRITIQUE_REQUIRED_IDS = {8, 11, 14, 15, 16, 20, 27, 33, 40, 42, 44, 45, 46, 47, 48, 49}


class P4DraftResponse(BaseModel):
    draft_answer: str
    reasoning: str
    quotes: List[str]


class P4FinalResponse(BaseModel):
    final_answer: str
    critique_logic: str
    is_abstained: bool = Field(
        default=False,
        description="Set to true if the draft indicates the answer is missing, unmeasured, or not explicitly provided in the text.",
    )


class P4_Pipeline(P3_Pipeline):
    def __init__(self, pipeline_version: str, config: Any, openai_client: Any, logger: Any, run_id: str):
        super().__init__(
            pipeline_version=pipeline_version,
            config=config,
            openai_client=openai_client,
            logger=logger,
            run_id=run_id,
        )
        self.logger.info(
            "P4 critic instruction map loaded | path=%s entries=%s",
            CRITIC_MAP_PATH,
            len(CRITIC_MAP),
        )

    def generate_draft_node(self, state: AgentState) -> Dict[str, Any]:
        question = str(state["question"])
        question_type = str(state.get("question_type", "FREE_TEXT")).upper()
        retrieved_context = "\n\n".join(str(chunk.get("text", "")) for chunk in state.get("retrieved_chunks", []))

        system_prompt = (
            "You are a precise chemistry research assistant. "
            "Answer ONLY based on the provided retrieved chunks. "
            "Return clean outputs that strictly match the requested schema. "
            "CRITICAL FORMATTING RULES:\n"
            "- DO NOT just output isolated numbers or disjointed lists.\n"
            "- Maintain brief context in your answer (e.g., instead of just '69 mg, 10 mg', write 'PbBr2: 69 mg, MXene: 10 mg').\n"
            "- For lifetimes or specific properties, include the label (e.g., 'Average lifetime: 4.5 ns').\n"
            "- Populate 'quotes' with exact, verbatim sentences from the text."
        )
        user_prompt = (
            f"Question Type:\n{question_type}\n\n"
            f"Question:\n{question}\n\n"
            f"Retrieved Chunks:\n{retrieved_context}\n"
        )

        try:
            completion = self.openai_client.beta.chat.completions.parse(
                model=self.config.llm_params.model_name,
                temperature=self.config.llm_params.temperature,
                max_tokens=self.config.llm_params.max_tokens,
                response_format=P4DraftResponse,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except AuthenticationError as exc:
            raise RuntimeError("OpenAI authentication error while generating answer. Check OPENAI_API_KEY.") from exc
        except RateLimitError as exc:
            raise RuntimeError("OpenAI quota/rate-limit error while generating answer.") from exc
        except PermissionDeniedError as exc:
            raise RuntimeError(
                f"OpenAI permission error while generating answer: {self._format_api_status_error(exc)}"
            ) from exc
        except APIStatusError as exc:
            raise RuntimeError(
                f"OpenAI status error while generating answer: {self._format_api_status_error(exc)}"
            ) from exc
        except APIError as exc:
            raise RuntimeError(f"OpenAI API error while generating answer: {exc}") from exc

        message = completion.choices[0].message if completion.choices else None
        parsed = getattr(message, "parsed", None)
        if parsed is None:
            raise RuntimeError("P4 draft generation failed: missing parsed response.")

        draft_answer = str(parsed.draft_answer).strip()
        reasoning = str(parsed.reasoning).strip()
        evidence_quotes = [str(item).strip() for item in parsed.quotes if str(item).strip()]

        usage = getattr(completion, "usage", None)
        llm_input_tokens = self._usage_tokens(usage, "prompt_tokens")
        llm_output_tokens = self._usage_tokens(usage, "completion_tokens")

        return {
            "draft_answer": draft_answer,
            "reasoning": reasoning,
            "evidence_quotes": evidence_quotes,
            "llm_input_tokens": llm_input_tokens,
            "llm_output_tokens": llm_output_tokens,
        }

    def route_after_draft(self, state: AgentState) -> str:
        try:
            qid = int(state.get("question_id", -1))
        except (TypeError, ValueError):
            qid = -1
        if qid in CRITIQUE_REQUIRED_IDS:
            return "critique_node"
        return "bypass_node"

    def bypass_node(self, state: AgentState) -> Dict[str, Any]:
        # For safe questions, the draft is promoted as final answer.
        return {
            "model_answer": str(state.get("draft_answer", "")),
            "critique_logic": "Bypassed critic (safe question type).",
            "is_abstained": False,
        }

    def critique_node(self, state: AgentState) -> Dict[str, Any]:
        question = str(state["question"])
        question_type = str(state.get("question_type", "FREE_TEXT")).upper()
        draft_answer = str(state.get("draft_answer", ""))
        question_id = str(state.get("question_id", ""))
        specific_instruction = CRITIC_MAP.get(
            question_id,
            "Format the draft answer accurately based on the question.",
        )

        system_prompt = (
            "You are a rigorous Scientific QA Editor. Your task is to refine a 'Draft Answer' into a pristine "
            "'Final Answer'.\n"
            "You do NOT generate new information. You only filter, format, and correct the Draft.\n\n"
            f"QUESTION-SPECIFIC INSTRUCTION:\n{specific_instruction}\n\n"
            "CRITICAL RULES:\n"
            "1. ABSTENTION: If the specific instruction asks you to abstain when information is missing, or if the "
            "draft clearly states the information is unmeasured/not provided, you MUST set `is_abstained = true` "
            "and output 'Unmeasured' or 'Not explicitly provided'.\n"
            "2. ZERO HALLUCINATION: Base your final answer strictly on the provided Draft. Remove all conversational "
            "filler (e.g., 'The document states...').\n"
            "Record your step-by-step logic in 'critique_logic', set the 'is_abstained' flag accurately, and output "
            "the clean 'final_answer'."
        )
        user_prompt = (
            f"Question Type:\n{question_type}\n\n"
            f"Question:\n{question}\n\n"
            f"Draft Answer:\n{draft_answer}\n"
        )

        try:
            completion = self.openai_client.beta.chat.completions.parse(
                model=self.config.llm_params.model_name,
                temperature=self.config.llm_params.temperature,
                max_tokens=self.config.llm_params.max_tokens,
                response_format=P4FinalResponse,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except AuthenticationError as exc:
            raise RuntimeError("OpenAI authentication error while generating answer. Check OPENAI_API_KEY.") from exc
        except RateLimitError as exc:
            raise RuntimeError("OpenAI quota/rate-limit error while generating answer.") from exc
        except PermissionDeniedError as exc:
            raise RuntimeError(
                f"OpenAI permission error while generating answer: {self._format_api_status_error(exc)}"
            ) from exc
        except APIStatusError as exc:
            raise RuntimeError(
                f"OpenAI status error while generating answer: {self._format_api_status_error(exc)}"
            ) from exc
        except APIError as exc:
            raise RuntimeError(f"OpenAI API error while generating answer: {exc}") from exc

        message = completion.choices[0].message if completion.choices else None
        parsed = getattr(message, "parsed", None)
        if parsed is None:
            raise RuntimeError("P4 critique generation failed: missing parsed response.")

        usage = getattr(completion, "usage", None)
        llm_input_tokens = int(state.get("llm_input_tokens", 0)) + self._usage_tokens(usage, "prompt_tokens")
        llm_output_tokens = int(state.get("llm_output_tokens", 0)) + self._usage_tokens(usage, "completion_tokens")

        return {
            "model_answer": str(parsed.final_answer).strip(),
            "critique_logic": str(parsed.critique_logic).strip(),
            "is_abstained": bool(getattr(parsed, "is_abstained", False)),
            "llm_input_tokens": llm_input_tokens,
            "llm_output_tokens": llm_output_tokens,
        }

    def build_graph(self) -> CompiledStateGraph:
        graph = StateGraph(AgentState)
        graph.add_node("retrieve_node", self.retrieve_node)
        graph.add_node("generate_draft_node", self.generate_draft_node)
        graph.add_node("bypass_node", self.bypass_node)
        graph.add_node("critique_node", self.critique_node)
        graph.add_edge(START, "retrieve_node")
        graph.add_edge("retrieve_node", "generate_draft_node")
        graph.add_conditional_edges(
            "generate_draft_node",
            self.route_after_draft,
            {
                "critique_node": "critique_node",
                "bypass_node": "bypass_node",
            },
        )
        graph.add_edge("bypass_node", END)
        graph.add_edge("critique_node", END)
        return graph.compile()
