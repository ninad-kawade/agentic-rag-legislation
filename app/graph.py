"""LangGraph corrective RAG loop with LLM relevance grading and abstention."""

from __future__ import annotations

import os
import re
from typing import Any, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from app.rerank import rerank
from app.retrieval import retrieve_hybrid
from app.schemas import Answer, Citation

load_dotenv()

MAX_RETRIES = 2
RELEVANCE_THRESHOLD = 1
GRADE_MODEL = os.getenv("RAG_GRADE_MODEL", "gpt-4o-mini")
GENERATE_MODEL = os.getenv("RAG_GENERATE_MODEL", "gpt-4o-mini")


class LLMQuotaError(RuntimeError):
    """Provider rate or quota limit; surfaced so the UI can show a busy message."""


def _raise_if_quota_error(exc: Exception) -> None:
    response = getattr(exc, "response", None)
    status_code = getattr(exc, "status_code", None) or getattr(response, "status_code", None)
    code = str(getattr(exc, "code", "")).lower()
    message = str(exc).lower()
    if status_code == 429 or "resource_exhausted" in code or "quota" in message and "429" in message:
        raise LLMQuotaError("LLM provider quota or rate limit reached") from exc


class AgentState(TypedDict, total=False):
    question: str
    query: str
    retries: int
    query_history: list[str]
    candidates: list[dict[str, Any]]
    documents: list[dict[str, Any]]
    grades: list[dict[str, Any]]
    enough_evidence: bool
    answer: dict[str, Any]
    abstained: bool
    llm_errors: list[str]


class DocumentGrade(BaseModel):
    index: int = Field(description="One-based document number from the supplied context")
    relevant: bool
    reason: str


class GradeOutput(BaseModel):
    grades: list[DocumentGrade]


class RewriteOutput(BaseModel):
    query: str


class GenerationOutput(BaseModel):
    text: str
    citation_indices: list[int]
    confidence: float = Field(ge=0.0, le=1.0)


def _chat_model(model_name: str) -> ChatOpenAI:
    """Create a chat model from configured OpenAI, Groq, or Gemini credentials."""
    load_dotenv()
    if os.getenv("OPENAI_API_KEY"):
        return ChatOpenAI(model=model_name, temperature=0, timeout=60, max_retries=0)
    if os.getenv("GROQ_API_KEY"):
        return ChatOpenAI(
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            base_url="https://api.groq.com/openai/v1",
            api_key=os.environ["GROQ_API_KEY"],
            temperature=0,
            timeout=60,
            max_retries=0,
        )
    if os.getenv("GOOGLE_API_KEY"):
        # Gemini's OpenAI-compatible endpoint keeps the project on the existing
        # OpenAI client dependency when only a Google key is configured.
        return ChatOpenAI(
            model=os.getenv("GOOGLE_CHAT_MODEL", "gemini-3.6-flash"),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=os.environ["GOOGLE_API_KEY"],
            temperature=0,
            extra_body={
                "extra_body": {
                    "google": {"thinking_config": {"thinking_level": "low"}}
                }
            },
            timeout=60,
            max_retries=0,
        )
    raise RuntimeError(
        "Configure OPENAI_API_KEY, GROQ_API_KEY, or GOOGLE_API_KEY to run the agent."
    )


def _response_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text", "")) if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content)


def _parse_json(model: type[BaseModel], content: Any) -> BaseModel:
    raw = _response_text(content).strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", raw, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        raw = fenced.group(1)
    return model.model_validate_json(raw)


def _context_block(documents: list[dict[str, Any]]) -> str:
    return "\n\n".join(
        f"[Document {index}] Source: {doc.get('source', 'unknown')}\n{doc.get('text', '')}"
        for index, doc in enumerate(documents, start=1)
    )


def retrieve_node(state: AgentState) -> AgentState:
    candidates = retrieve_hybrid(state["query"], candidate_k=50, top_k=20, rrf_k=60)
    return {"candidates": candidates}


def rerank_node(state: AgentState) -> AgentState:
    documents = rerank(state["query"], state.get("candidates", []), top_k=5)
    return {"documents": documents}


def grade_docs_node(state: AgentState) -> AgentState:
    documents = state.get("documents", [])
    if not documents:
        return {"grades": [], "enough_evidence": False}

    model = _chat_model(GRADE_MODEL)
    prompt = (
        "Judge each document only for whether it contains evidence that can help answer "
        "the question. A shared topic or matching entity alone is not enough. Mark unrelated "
        "documents false. Return one grade for every numbered document, preserving its index. "
        "Return only JSON matching {\"grades\":[{\"index\":1,\"relevant\":true,\"reason\":\"...\"}]}\n\n"
        f"Question: {state['question']}\n\n{_context_block(documents)}"
    )
    try:
        result = _parse_json(GradeOutput, model.invoke([
            SystemMessage(content="You are a strict relevance grader for legal retrieval."),
            HumanMessage(content=prompt),
        ]).content)
        grades_by_index = {grade.index: grade for grade in result.grades}
        grades = [
            {
                "index": index,
                "relevant": bool(grades_by_index[index].relevant) if index in grades_by_index else False,
                "reason": grades_by_index[index].reason if index in grades_by_index else "No valid grade returned.",
            }
            for index in range(1, len(documents) + 1)
        ]
    except Exception as exc:
        _raise_if_quota_error(exc)
        # Fail closed: model/API/parse errors must never turn weak retrieval into
        # an unsupported answer.
        grades = [
            {"index": index, "relevant": False, "reason": f"Grading failed: {type(exc).__name__}"}
            for index in range(1, len(documents) + 1)
        ]
    relevant_count = sum(grade["relevant"] for grade in grades)
    update: AgentState = {"grades": grades, "enough_evidence": relevant_count >= RELEVANCE_THRESHOLD}
    if "exc" in locals():
        update["llm_errors"] = [*state.get("llm_errors", []), f"grade_docs:{type(exc).__name__}"]
    return update


def generate_node(state: AgentState) -> AgentState:
    documents = state.get("documents", [])
    grades = state.get("grades", [])
    relevant_docs = [
        doc for index, doc in enumerate(documents, start=1)
        if index <= len(grades) and grades[index - 1]["relevant"]
    ]
    if not relevant_docs:
        return abstain_node(state)

    prompt = (
        "Answer only from the provided context. If the context does not contain the answer, "
        "say you cannot answer from the available documents. Do not use outside knowledge or "
        "guess. Return only JSON with keys text, citation_indices, confidence. citation_indices "
        "must be one-based document numbers from the supplied context. Use at least one citation "
        "for a factual answer.\n\n"
        f"Question: {state['question']}\n\n{_context_block(relevant_docs)}"
    )
    model = _chat_model(GENERATE_MODEL)
    try:
        generated = _parse_json(GenerationOutput, model.invoke([
            SystemMessage(content="You are a careful legal research assistant. Ground every claim in supplied text."),
            HumanMessage(content=prompt),
        ]).content)
        valid_indices = sorted({
            index for index in generated.citation_indices if 1 <= index <= len(relevant_docs)
        })
        if not valid_indices:
            return abstain_node(state)
        citations = [
            Citation(
                source=str(relevant_docs[index - 1].get("source", "unknown")),
                snippet=str(relevant_docs[index - 1].get("text", ""))[:500],
                source_id=relevant_docs[index - 1].get("chunk_id"),
            )
            for index in valid_indices
        ]
        answer = Answer(
            text=generated.text,
            citations=citations,
            confidence=generated.confidence,
        )
        return {"answer": answer.model_dump(), "abstained": False}
    except Exception as exc:
        _raise_if_quota_error(exc)
        # Invalid structured output or provider errors fail closed.
        update = abstain_node(state)
        update["llm_errors"] = [*state.get("llm_errors", []), f"generate:{type(exc).__name__}"]
        return update


def rewrite_query_node(state: AgentState) -> AgentState:
    model = _chat_model(GRADE_MODEL)
    prompt = (
        "Rewrite the search query to improve retrieval in a corpus of UK and EU legislation. "
        "Keep named instruments, numbers, dates, and entities. Do not answer the question. "
        "Return only JSON: {\"query\":\"...\"}.\n\n"
        f"Original question: {state['question']}\n"
        f"Previous query: {state['query']}\n"
        f"Attempt: {state.get('retries', 0) + 1} of {MAX_RETRIES}"
    )
    try:
        rewritten = _parse_json(RewriteOutput, model.invoke([
            SystemMessage(content="You rewrite search queries; output valid JSON only."),
            HumanMessage(content=prompt),
        ]).content)
        new_query = rewritten.query.strip()
        if not new_query:
            new_query = state["query"]
    except Exception as exc:
        _raise_if_quota_error(exc)
        new_query = state["query"]
        llm_errors = [*state.get("llm_errors", []), f"rewrite_query:{type(exc).__name__}"]
    else:
        llm_errors = state.get("llm_errors", [])
    history = [*state.get("query_history", []), new_query]
    return {
        "query": new_query,
        "query_history": history,
        "retries": state.get("retries", 0) + 1,
        "llm_errors": llm_errors,
    }


def abstain_node(state: AgentState) -> AgentState:
    answer = Answer(
        text="Insufficient evidence to answer from the available documents.",
        citations=[],
        confidence=0.0,
    )
    return {"answer": answer.model_dump(), "abstained": True}


def _after_grading(state: AgentState) -> str:
    if state.get("enough_evidence", False):
        return "generate"
    if state.get("retries", 0) < MAX_RETRIES:
        return "rewrite_query"
    return "abstain"


def build_graph():
    """Build and compile the retrieve → rerank → grade → answer/retry graph."""
    graph = StateGraph(AgentState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("grade_docs", grade_docs_node)
    graph.add_node("generate", generate_node)
    graph.add_node("rewrite_query", rewrite_query_node)
    graph.add_node("abstain", abstain_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "grade_docs")
    graph.add_conditional_edges(
        "grade_docs",
        _after_grading,
        {"generate": "generate", "rewrite_query": "rewrite_query", "abstain": "abstain"},
    )
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("generate", END)
    graph.add_edge("abstain", END)
    return graph.compile()


def ask(question: str, *, graph=None) -> dict[str, Any]:
    """Run the agent and return its validated answer plus abstention metadata."""
    if not question.strip():
        raise ValueError("question must not be empty")
    compiled = graph or build_graph()
    result = compiled.invoke({
        "question": question.strip(),
        "query": question.strip(),
        "retries": 0,
        "query_history": [question.strip()],
    })
    result["answer"] = Answer.model_validate(result["answer"])
    result["abstained"] = bool(result.get("abstained", False))
    return result
