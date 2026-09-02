"""Custom exception classes for the RAG application."""


class RAGException(Exception):
    """Base exception for all RAG-related errors."""

    def __init__(self, message: str, error_code: str = None, details: dict = None):
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        super().__init__(self.message)

    def to_dict(self):
        """Convert exception to dictionary for API responses."""
        return {
            "error": self.error_code,
            "message": self.message,
            "details": self.details,
        }


# ============================================================================
# File Handling Exceptions
# ============================================================================


class FileHandlingException(RAGException):
    """Base exception for file handling errors."""

    pass


class FileTooLargeException(FileHandlingException):
    """Raised when uploaded file exceeds size limit."""

    def __init__(self, filename: str, size_mb: float, max_size_mb: int):
        super().__init__(
            message=f"File '{filename}' ({size_mb:.1f}MB) exceeds maximum size of {max_size_mb}MB",
            error_code="FILE_TOO_LARGE",
            details={
                "filename": filename,
                "size_mb": size_mb,
                "max_size_mb": max_size_mb,
            },
        )


class UnsupportedFileFormatException(FileHandlingException):
    """Raised when file format is not supported."""

    def __init__(self, filename: str, supported_formats: list):
        super().__init__(
            message=f"File format not supported. Supported formats: {', '.join(supported_formats)}",
            error_code="UNSUPPORTED_FORMAT",
            details={
                "filename": filename,
                "supported_formats": supported_formats,
            },
        )


class FileProcessingException(FileHandlingException):
    """Raised when file processing fails."""

    def __init__(self, filename: str, reason: str):
        super().__init__(
            message=f"Failed to process file '{filename}': {reason}",
            error_code="FILE_PROCESSING_ERROR",
            details={"filename": filename, "reason": reason},
        )


# ============================================================================
# LLM & API Exceptions
# ============================================================================


class LLMException(RAGException):
    """Base exception for LLM-related errors."""

    pass


class APIKeyMissingException(LLMException):
    """Raised when required API key is missing."""

    def __init__(self, provider: str):
        super().__init__(
            message=f"{provider} API key not found. Set {provider}_API_KEY environment variable.",
            error_code="API_KEY_MISSING",
            details={"provider": provider},
        )


class LLMTimeoutException(LLMException):
    """Raised when LLM request times out."""

    def __init__(self, timeout_seconds: int):
        super().__init__(
            message=f"LLM request timed out after {timeout_seconds} seconds",
            error_code="LLM_TIMEOUT",
            details={"timeout_seconds": timeout_seconds},
        )


class LLMRateLimitException(LLMException):
    """Raised when API rate limit is exceeded."""

    def __init__(self, retry_after: int = None):
        super().__init__(
            message="LLM API rate limit exceeded. Please try again later.",
            error_code="RATE_LIMIT_EXCEEDED",
            details={"retry_after_seconds": retry_after},
        )


class ModelUnavailableException(LLMException):
    """Raised when requested model is not available."""

    def __init__(self, model_name: str, fallback_model: str = None):
        msg = f"Model '{model_name}' is not available"
        details = {"model_name": model_name}
        if fallback_model:
            msg += f". Falling back to '{fallback_model}'"
            details["fallback_model"] = fallback_model
        super().__init__(
            message=msg,
            error_code="MODEL_UNAVAILABLE",
            details=details,
        )


# ============================================================================
# Retrieval & Vectorstore Exceptions
# ============================================================================


class VectorstoreException(RAGException):
    """Base exception for vectorstore-related errors."""

    pass


class VectorstoreNotFoundException(VectorstoreException):
    """Raised when vectorstore doesn't exist for workspace."""

    def __init__(self, workspace_id: str):
        super().__init__(
            message=f"Vectorstore not found for workspace '{workspace_id}'",
            error_code="VECTORSTORE_NOT_FOUND",
            details={"workspace_id": workspace_id},
        )


class RetrievalFailedException(VectorstoreException):
    """Raised when document retrieval fails."""

    def __init__(self, reason: str, query: str = None):
        details = {"reason": reason}
        if query:
            details["query"] = query[:100]  # Truncate long queries
        super().__init__(
            message=f"Document retrieval failed: {reason}",
            error_code="RETRIEVAL_FAILED",
            details=details,
        )


# ============================================================================
# Workspace & Storage Exceptions
# ============================================================================


class WorkspaceException(RAGException):
    """Base exception for workspace-related errors."""

    pass


class WorkspaceNotFoundException(WorkspaceException):
    """Raised when workspace doesn't exist."""

    def __init__(self, workspace_id: str):
        super().__init__(
            message=f"Workspace '{workspace_id}' not found",
            error_code="WORKSPACE_NOT_FOUND",
            details={"workspace_id": workspace_id},
        )


class WorkspaceAccessException(WorkspaceException):
    """Raised when user lacks permission to access workspace."""

    def __init__(self, workspace_id: str, user_id: str):
        super().__init__(
            message=f"Access denied to workspace '{workspace_id}'",
            error_code="WORKSPACE_ACCESS_DENIED",
            details={"workspace_id": workspace_id, "user_id": user_id},
        )


class InvalidWorkspaceNameException(WorkspaceException):
    """Raised when workspace name is invalid."""

    def __init__(self, name: str, reason: str = ""):
        msg = f"Invalid workspace name: '{name}'"
        if reason:
            msg += f". {reason}"
        super().__init__(
            message=msg,
            error_code="INVALID_WORKSPACE_NAME",
            details={"name": name, "reason": reason},
        )


# ============================================================================
# Input Validation Exceptions
# ============================================================================


class ValidationException(RAGException):
    """Base exception for input validation errors."""

    pass


class InvalidQueryException(ValidationException):
    """Raised when query is invalid."""

    def __init__(self, reason: str):
        super().__init__(
            message=f"Invalid query: {reason}",
            error_code="INVALID_QUERY",
            details={"reason": reason},
        )


class InvalidParameterException(ValidationException):
    """Raised when parameter value is invalid."""

    def __init__(self, parameter: str, value: str, expected: str):
        super().__init__(
            message=f"Invalid parameter '{parameter}': {value}. Expected: {expected}",
            error_code="INVALID_PARAMETER",
            details={
                "parameter": parameter,
                "value": str(value),
                "expected": expected,
            },
        )


# ============================================================================
# Configuration Exceptions
# ============================================================================


class ConfigurationException(RAGException):
    """Base exception for configuration errors."""

    pass


class MissingConfigException(ConfigurationException):
    """Raised when required configuration is missing."""

    def __init__(self, config_key: str):
        super().__init__(
            message=f"Required configuration '{config_key}' is missing",
            error_code="MISSING_CONFIG",
            details={"config_key": config_key},
        )


class InvalidConfigException(ConfigurationException):
    """Raised when configuration value is invalid."""

    def __init__(self, config_key: str, value: str, reason: str):
        super().__init__(
            message=f"Invalid configuration '{config_key}': {reason}",
            error_code="INVALID_CONFIG",
            details={
                "config_key": config_key,
                "value": str(value),
                "reason": reason,
            },
        )


# ============================================================================
# Evaluation Exceptions
# ============================================================================


class EvaluationException(RAGException):
    """Base exception for evaluation-related errors."""

    pass


class DatasetNotFoundException(EvaluationException):
    """Raised when evaluation dataset not found."""

    def __init__(self, workspace_id: str):
        super().__init__(
            message=f"Evaluation dataset not found for workspace '{workspace_id}'",
            error_code="DATASET_NOT_FOUND",
            details={"workspace_id": workspace_id},
        )


class EvaluationRunException(EvaluationException):
    """Raised when evaluation run fails."""

    def __init__(self, reason: str, run_id: str = None):
        details = {"reason": reason}
        if run_id:
            details["run_id"] = run_id
        super().__init__(
            message=f"Evaluation run failed: {reason}",
            error_code="EVALUATION_RUN_FAILED",
            details=details,
        )


# ============================================================================
# Database Exceptions
# ============================================================================


class DatabaseException(RAGException):
    """Base exception for database-related errors."""

    pass


class DatabaseConnectionException(DatabaseException):
    """Raised when database connection fails."""

    def __init__(self, reason: str):
        super().__init__(
            message=f"Database connection failed: {reason}",
            error_code="DB_CONNECTION_ERROR",
            details={"reason": reason},
        )


class DatabaseQueryException(DatabaseException):
    """Raised when database query fails."""

    def __init__(self, query: str, reason: str):
        super().__init__(
            message=f"Database query failed: {reason}",
            error_code="DB_QUERY_ERROR",
            details={"reason": reason, "query": query[:100]},
        )
