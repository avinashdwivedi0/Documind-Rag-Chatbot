"""Pydantic models for request/response validation and type safety."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, validator

# ============================================================================
# Enums
# ============================================================================


class ResponseMode(str, Enum):
    """Available response modes for the RAG chain."""

    DEEP_RESEARCH = "Deep research"
    EXECUTIVE_BRIEF = "Executive brief"
    COMPARE_VIEWPOINTS = "Compare viewpoints"
    STUDY_GUIDE = "Study guide"
    ACTION_PLAN = "Action plan"


class EvaluatorMode(str, Enum):
    """Available evaluator modes."""

    RULE_BASED = "rule"
    LLM_BASED = "llm"


class Environment(str, Enum):
    """Application environments."""

    DEV = "dev"
    STAGING = "staging"
    PROD = "prod"


# ============================================================================
# Request Models
# ============================================================================


class QueryRequest(BaseModel):
    """Request model for asking a question about documents."""

    question: str = Field(..., min_length=1, max_length=2000, description="The question to ask")
    workspace_id: str = Field(..., description="ID of the workspace to query")
    conversation_id: str = Field(default="default", description="Conversation ID for context")
    response_mode: ResponseMode = Field(default=ResponseMode.DEEP_RESEARCH)
    top_k: int = Field(default=3, ge=1, le=20, description="Number of documents to retrieve")

    @field_validator("question")
    @classmethod
    def question_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Question cannot be empty or whitespace only")
        return v.strip()

    @field_validator("workspace_id")
    @classmethod
    def workspace_id_valid(cls, v):
        if not v.replace("-", "").replace("_", "").isalnum():
            raise ValueError(
                "Workspace ID must contain only alphanumeric characters, hyphens, and underscores"
            )
        return v


class CreateWorkspaceRequest(BaseModel):
    """Request model for creating a new workspace."""

    name: str = Field(..., min_length=1, max_length=100, description="Workspace name")
    description: Optional[str] = Field(None, max_length=500)

    @field_validator("name")
    @classmethod
    def name_valid(cls, v):
        if not v.strip():
            raise ValueError("Workspace name cannot be empty")
        return v.strip()


class AddCaseRequest(BaseModel):
    """Request model for adding an evaluation case."""

    question: str = Field(..., min_length=1, max_length=2000)
    expected_answer: Optional[str] = Field(None, max_length=5000)
    expected_sources: Optional[List[str]] = Field(None)
    expected_pages: Optional[List[int]] = Field(None)


class RunEvaluationRequest(BaseModel):
    """Request model for running evaluation on dataset."""

    workspace_id: str
    top_k: int = Field(default=3, ge=1, le=20)
    evaluator_mode: EvaluatorMode = Field(default=EvaluatorMode.RULE_BASED)


# ============================================================================
# Response Models
# ============================================================================


class SourceDocument(BaseModel):
    """Model for a source document reference."""

    source: str = Field(..., description="File name or document ID")
    page: Optional[int] = Field(None, description="Page number (0-indexed)")
    excerpt: str = Field(..., description="Relevant text excerpt")
    relevance_score: Optional[float] = Field(None, ge=0.0, le=1.0)


class QueryResponse(BaseModel):
    """Response model for a query."""

    answer: str = Field(..., description="The generated answer")
    sources: List[SourceDocument] = Field(default_factory=list)
    interpreted_query: Optional[str] = Field(
        None, description="Expanded/interpreted query used for retrieval"
    )
    conversation_id: str
    workspace_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (tokens, latency, etc.)",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "answer": "Based on the documents...",
                "sources": [
                    {
                        "source": "report.pdf",
                        "page": 5,
                        "excerpt": "The key finding is...",
                        "relevance_score": 0.95,
                    }
                ],
                "interpreted_query": "What is the main topic about...",
                "conversation_id": "default",
                "workspace_id": "ws-123",
                "metadata": {
                    "tokens": 256,
                    "latency_ms": 2340,
                    "model": "llama-3.3-70b-versatile",
                },
            }
        }


class WorkspaceInfo(BaseModel):
    """Information about a workspace."""

    workspace_id: str
    name: str
    description: Optional[str] = None
    document_count: int = 0
    created_at: datetime
    updated_at: datetime
    size_mb: Optional[float] = None
    conversations: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ConversationInfo(BaseModel):
    """Information about a conversation."""

    conversation_id: str
    workspace_id: str
    created_at: datetime
    updated_at: datetime
    message_count: int = 0
    last_message: Optional[str] = None


class EvaluationMetrics(BaseModel):
    """Evaluation metrics for a case."""

    hit: bool = Field(..., description="Whether expected source was retrieved in top-k")
    rank: Optional[int] = Field(None, description="Rank of expected source (1-indexed)")
    recall_at_k: Optional[int] = Field(None, description="Recall@k score")
    precision_at_k: Optional[float] = Field(None, description="Precision@k score")
    mrr: Optional[float] = Field(None, description="Mean Reciprocal Rank")
    avg_retrieval_score: Optional[float] = Field(
        None, description="Average retrieval relevance score"
    )


class EvaluationResult(BaseModel):
    """Result of evaluating a single case."""

    case_id: str
    question: str
    metrics: EvaluationMetrics
    answer_score: Optional[float] = Field(None, description="Answer similarity score")
    citation_match_rate: Optional[float] = Field(None, description="Citation accuracy")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class EvaluationRunResult(BaseModel):
    """Result of an evaluation run."""

    run_id: str
    workspace_id: str
    created_at: datetime
    total_cases: int
    passed_cases: int
    hit_rate: float = Field(..., ge=0.0, le=1.0)
    avg_mrr: Optional[float] = None
    avg_latency_ms: Optional[int] = None
    results: List[EvaluationResult] = Field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        """Calculate pass rate."""
        return self.passed_cases / self.total_cases if self.total_cases > 0 else 0.0


# ============================================================================
# Configuration Models
# ============================================================================


class LLMConfig(BaseModel):
    """LLM configuration."""

    api_key: str = Field(..., description="API key for LLM provider")
    model_name: str = Field(default="openai/gpt-oss-20b")
    max_tokens: int = Field(default=1024, ge=1, le=4096)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    timeout_seconds: int = Field(default=60, ge=1, le=600)

    class Config:
        # Don't expose API key in logs/debug
        underscore_attrs_are_private = True


class RetrievalConfig(BaseModel):
    """Retrieval configuration."""

    top_k: int = Field(default=3, ge=1, le=20, description="Number of documents to retrieve")
    candidate_k: int = Field(
        default=8, ge=1, le=50, description="Number of candidates for hybrid search"
    )
    context_truncation_chars: int = Field(default=2048, ge=256, le=10000)
    similarity_threshold: float = Field(default=0.0, ge=0.0, le=1.0)

    @field_validator("candidate_k")
    @classmethod
    def candidate_k_ge_top_k(cls, v, info):
        if "top_k" in info.data and v < info.data["top_k"]:
            raise ValueError("candidate_k must be >= top_k")
        return v


class EvaluationConfig(BaseModel):
    """Evaluation configuration."""

    mode: EvaluatorMode = Field(default=EvaluatorMode.RULE_BASED)
    token_cost_per_1k: Optional[float] = Field(None, description="Cost per 1000 tokens in USD")
    llm_provider: str = Field(default="unknown")
    enable_tracing: bool = Field(default=False)


class StorageConfig(BaseModel):
    """Storage configuration."""

    upload_dir: str = Field(default="./data/uploads")
    vector_dir: str = Field(default="./data/vectors")
    history_dir: str = Field(default="./data/chat_history")
    eval_dir: str = Field(default="./data/evaluations")
    max_file_size_mb: int = Field(default=25, ge=1, le=1000)


class ApplicationConfig(BaseModel):
    """Overall application configuration."""

    environment: Environment = Field(default=Environment.DEV)
    log_level: str = Field(default="INFO")
    llm: LLMConfig
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    enable_redis_cache: bool = Field(default=False)
    enable_batch_processing: bool = Field(default=False)
    enable_api_logging: bool = Field(default=True)

    class Config:
        validate_assignment = True


# ============================================================================
# Error Response Models
# ============================================================================


class ErrorDetail(BaseModel):
    """Details of an error."""

    error: str = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    details: Dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    """Error response model."""

    status: int = Field(..., ge=400, le=599)
    error: ErrorDetail
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_schema_extra = {
            "example": {
                "status": 400,
                "error": {
                    "error": "INVALID_QUERY",
                    "message": "Invalid query: Question cannot be empty",
                    "details": {"reason": "Empty question provided"},
                },
                "timestamp": "2026-09-01T10:30:00Z",
            }
        }


# ============================================================================
# Trace & Monitoring Models
# ============================================================================


class QueryTrace(BaseModel):
    """Trace information for a query."""

    trace_id: str = Field(..., description="Unique trace ID")
    workspace_id: str
    question: str = Field(..., max_length=256)
    retrieval_ms: int = Field(..., ge=0, description="Time spent on retrieval")
    llm_ms: int = Field(..., ge=0, description="Time spent on LLM inference")
    total_ms: int = Field(..., ge=0, description="Total latency")
    retrieved_chunks: int = Field(..., ge=0)
    retrieval_top_k: int
    tokens: Dict[str, Any] = Field(default_factory=dict)
    cost: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
