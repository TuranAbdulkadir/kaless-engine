"""KALESS Engine — Custom Exception Classes."""


class KalessEngineError(Exception):
    """Base exception for KALESS engine errors."""

    def __init__(self, message: str, code: str = "ENGINE_ERROR"):
        self.message = message
        self.code = code
        super().__init__(self.message)


class ParseError(KalessEngineError):
    """Raised when dataset parsing fails."""

    def __init__(self, message: str):
        super().__init__(message, code="PARSE_ERROR")


class ValidationError(KalessEngineError):
    """Raised when input validation fails."""

    def __init__(self, message: str):
        super().__init__(message, code="VALIDATION_ERROR")


class AnalysisError(KalessEngineError):
    """Raised when a statistical analysis fails."""

    def __init__(self, message: str):
        super().__init__(message, code="ANALYSIS_ERROR")


class InsufficientDataError(KalessEngineError):
    """Raised when there is insufficient data for an analysis."""

    def __init__(self, message: str = "Insufficient data for this analysis"):
        super().__init__(message, code="INSUFFICIENT_DATA")


class AssumptionViolationWarning(KalessEngineError):
    """Raised when a statistical assumption is violated (non-fatal)."""

    def __init__(self, message: str):
        super().__init__(message, code="ASSUMPTION_VIOLATION")


class TransformError(KalessEngineError):
    """Raised when a data transformation fails."""

    def __init__(self, message: str):
        super().__init__(message, code="TRANSFORM_ERROR")


class ExportError(KalessEngineError):
    """Raised when export generation fails."""

    def __init__(self, message: str):
        super().__init__(message, code="EXPORT_ERROR")


class PlanLimitError(KalessEngineError):
    """Raised when a plan limit is exceeded."""

    def __init__(self, message: str = "Plan limit exceeded"):
        super().__init__(message, code="PLAN_LIMIT_EXCEEDED")
