# Backend Core Module

Professional-grade utilities, type definitions, and exception handling for the RAG application.

## Contents

### 1. Exception Classes (`exceptions.py`)

Custom, typed exception hierarchy for clear error handling and API responses.

#### Usage Example

```python
from backend.core.exceptions import (
    FileTooLargeException,
    APIKeyMissingException,
    RetrievalFailedException
)

# In file_handler.py
def upload_file(filepath, max_size_mb=25):
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    if size_mb > max_size_mb:
        raise FileTooLargeException(
            filename=os.path.basename(filepath),
            size_mb=size_mb,
            max_size_mb=max_size_mb
        )

# In rag_chain.py
def create_chat_chain():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise APIKeyMissingException(provider="Groq")
    # ...

# In query_service.py
def retrieve_documents(query, top_k=3):
    try:
        results = vectorstore.similarity_search(query, k=top_k)
    except Exception as e:
        raise RetrievalFailedException(
            reason=f"Vector search failed: {str(e)}",
            query=query
        )
```

#### Exception Hierarchy

```
RAGException (base)
├── FileHandlingException
│   ├── FileTooLargeException
│   ├── UnsupportedFileFormatException
│   └── FileProcessingException
├── LLMException
│   ├── APIKeyMissingException
│   ├── LLMTimeoutException
│   ├── LLMRateLimitException
│   └── ModelUnavailableException
├── VectorstoreException
│   ├── VectorstoreNotFoundException
│   └── RetrievalFailedException
├── WorkspaceException
│   ├── WorkspaceNotFoundException
│   ├── WorkspaceAccessException
│   └── InvalidWorkspaceNameException
├── ValidationException
│   ├── InvalidQueryException
│   └── InvalidParameterException
├── ConfigurationException
│   ├── MissingConfigException
│   └── InvalidConfigException
├── EvaluationException
│   ├── DatasetNotFoundException
│   └── EvaluationRunException
└── DatabaseException
    ├── DatabaseConnectionException
    └── DatabaseQueryException
```

#### Converting to API Response

All exceptions have a `to_dict()` method:

```python
try:
    # some operation
except RAGException as e:
    return {
        "status": 400,
        "error": e.to_dict(),
        "timestamp": datetime.utcnow().isoformat()
    }

# Output:
# {
#     "status": 400,
#     "error": {
#         "error": "FILE_TOO_LARGE",
#         "message": "File 'report.pdf' (28.5MB) exceeds maximum size of 25MB",
#         "details": {
#             "filename": "report.pdf",
#             "size_mb": 28.5,
#             "max_size_mb": 25
#         }
#     },
#     "timestamp": "2026-09-01T10:30:00.123456"
# }
```

---

### 2. Pydantic Models (`models.py`)

Type-safe request/response validation with automatic documentation.

#### Request Models

```python
from backend.core.models import QueryRequest, ResponseMode

# Type-safe request parsing
request = QueryRequest(
    question="What does the document say about X?",
    workspace_id="ws-123",
    conversation_id="conv-456",
    response_mode=ResponseMode.DEEP_RESEARCH,
    top_k=5
)

# Automatic validation - raises ValidationError if invalid
try:
    bad_request = QueryRequest(question="", workspace_id="")
except ValidationError as e:
    # FastAPI will automatically return 422 Unprocessable Entity
    print(e.json())
```

#### Response Models

```python
from backend.core.models import QueryResponse, SourceDocument
from datetime import datetime

response = QueryResponse(
    answer="Based on the documents, ...",
    sources=[
        SourceDocument(
            source="report.pdf",
            page=5,
            excerpt="The key finding is...",
            relevance_score=0.95
        )
    ],
    interpreted_query="Rephrased query for retrieval",
    conversation_id="conv-456",
    workspace_id="ws-123",
    metadata={
        "tokens": 256,
        "latency_ms": 2340,
        "model": "llama-3.1-8b-instant"
    }
)

# Serialize to JSON automatically
json_response = response.model_dump_json(indent=2)
```

#### Configuration Models

```python
from backend.core.models import ApplicationConfig, LLMConfig, RetrievalConfig

# Type-safe configuration with validation
config = ApplicationConfig(
    environment=Environment.DEV,
    llm=LLMConfig(
        api_key="sk-...",
        model_name="llama-3.1-8b-instant",
        max_tokens=1024,
        temperature=0.0
    ),
    retrieval=RetrievalConfig(
        top_k=3,
        candidate_k=8,
        context_truncation_chars=2048
    )
)

# Access as typed properties
assert config.llm.temperature == 0.0
assert config.retrieval.top_k == 3
```

#### Evaluation Models

```python
from backend.core.models import EvaluationResult, EvaluationRunResult, EvaluationMetrics

result = EvaluationRunResult(
    run_id="run-123",
    workspace_id="ws-123",
    total_cases=50,
    passed_cases=45,
    hit_rate=0.90,
    avg_mrr=0.75,
    avg_latency_ms=2300,
    results=[
        EvaluationResult(
            case_id="case-1",
            question="What is X?",
            metrics=EvaluationMetrics(
                hit=True,
                rank=1,
                recall_at_k=0.95,
                precision_at_k=0.92,
                mrr=1.0,
                avg_retrieval_score=0.94
            ),
            answer_score=0.88,
            citation_match_rate=1.0
        )
    ]
)

# Calculate metrics
assert result.pass_rate == 0.90  # Auto-calculated property
```

---

### 3. Using with FastAPI (Future)

When building a REST API, the models integrate seamlessly:

```python
from fastapi import FastAPI, HTTPException
from backend.core.models import QueryRequest, QueryResponse
from backend.core.exceptions import RAGException

app = FastAPI()

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Query documents with automatic validation."""
    try:
        answer = rag_chain.run(request.question)
        return QueryResponse(
            answer=answer,
            workspace_id=request.workspace_id,
            # ...
        )
    except RAGException as e:
        raise HTTPException(
            status_code=400,
            detail=e.to_dict()
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "INTERNAL_ERROR",
                "message": "An unexpected error occurred"
            }
        )
```

---

### 4. Enums for Type Safety

```python
from backend.core.models import ResponseMode, EvaluatorMode, Environment

# Prevent invalid string values
mode: ResponseMode = ResponseMode.DEEP_RESEARCH
assert mode == "Deep research"  # Enum value

# Works in string contexts
print(f"Mode: {mode.value}")  # Output: Mode: Deep research

# Type checking catches errors at development time
evaluator: EvaluatorMode = EvaluatorMode.LLM_BASED  # ✓ Valid
# evaluator = "invalid"  # ✗ Type error
```

---

## Integration Checklist

### Phase 1: Replace Exceptions (2-3 hours)

1. Update `backend/file_handler.py`:
   ```python
   from backend.core.exceptions import FileTooLargeException, UnsupportedFileFormatException
   # Replace: raise Exception("...")
   # With: raise FileTooLargeException(...)
   ```

2. Update `backend/rag_chain.py`:
   ```python
   from backend.core.exceptions import APIKeyMissingException, ModelUnavailableException
   ```

3. Update `backend/evaluation_service.py`:
   ```python
   from backend.core.exceptions import EvaluationRunException, DatasetNotFoundException
   ```

4. Update `backend/query_service.py`:
   ```python
   from backend.core.exceptions import RetrievalFailedException, InvalidQueryException
   ```

### Phase 2: Add Type Hints (3-4 hours)

1. Add imports to each module:
   ```python
   from typing import Dict, List, Optional, Tuple
   from backend.core.models import QueryResponse, SourceDocument
   ```

2. Update function signatures:
   ```python
   # Before
   def retrieve_documents(query, top_k=3):
       # ...

   # After
   def retrieve_documents(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
       # ...
   ```

### Phase 3: Add Docstrings (2-3 hours)

```python
def retrieve_documents(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """Retrieve similar documents from vectorstore.

    Performs similarity search using the query embedding against the
    FAISS index and returns top-k most relevant documents.

    Args:
        query: The search query string (max 2000 characters)
        top_k: Number of documents to retrieve (default: 3, max: 20)

    Returns:
        List of document dictionaries with keys:
        - source: Document filename or ID
        - page: Page number (0-indexed)
        - excerpt: Relevant text excerpt
        - score: Relevance score (0.0-1.0)

    Raises:
        RetrievalFailedException: If vectorstore search fails
        InvalidQueryException: If query is empty or too long

    Example:
        >>> docs = retrieve_documents("climate change", top_k=5)
        >>> len(docs)
        5
        >>> docs[0]["score"]
        0.95
    """
    if not query or not query.strip():
        raise InvalidQueryException("Query cannot be empty")
    if len(query) > 2000:
        raise InvalidQueryException("Query exceeds maximum length of 2000 characters")

    try:
        results = vectorstore.similarity_search(query.strip(), k=top_k)
        return [
            {
                "source": r.metadata.get("source", "unknown"),
                "page": r.metadata.get("page"),
                "excerpt": r.page_content,
                "score": r.metadata.get("score", 0.0)
            }
            for r in results
        ]
    except Exception as e:
        raise RetrievalFailedException(reason=str(e), query=query)
```

---

## Best Practices

### 1. Always Use Specific Exceptions
```python
# ❌ Don't
raise Exception("File too large")

# ✅ Do
raise FileTooLargeException("report.pdf", 28.5, 25)
```

### 2. Validate Input with Models
```python
# ❌ Don't
def query(question, workspace_id, top_k):
    if not question:
        raise Exception("Question is required")
    # ...

# ✅ Do
from backend.core.models import QueryRequest

def query(request: QueryRequest):
    # request is already validated
    # ...
```

### 3. Use Type Hints Everywhere
```python
# ❌ Don't
def get_config():
    return config

# ✅ Do
def get_config() -> ApplicationConfig:
    return config
```

### 4. Document with Docstrings
```python
def expensive_operation(data: List[str]) -> Dict[str, float]:
    """Compute expensive metrics on input data.

    This function performs heavy computation and should be cached.

    Args:
        data: List of text documents

    Returns:
        Dictionary mapping document ID to score

    Raises:
        ValueError: If data is empty
    """
    # ...
```

---

## Files Created

- `backend/core/__init__.py` - Module exports (already created)
- `backend/core/exceptions.py` - Exception classes (already created)
- `backend/core/models.py` - Pydantic models (already created)

## Files to Create Next

- `backend/core/logging.py` - Structured logging utilities
- `backend/core/config.py` - Configuration management
- `backend/core/utils.py` - Utility functions

---

## Testing

See `requirements-dev.txt` and `pytest.ini` for testing setup.

Run tests with:
```bash
pytest tests/ -v --cov=backend
```

---

## References

- [Pydantic Documentation](https://docs.pydantic.dev)
- [Python Type Hints](https://docs.python.org/3/library/typing.html)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [Exception Best Practices](https://docs.python.org/3/tutorial/errors.html)
