"""Pipeline-specific exceptions."""


class PipelineError(Exception):
    pass


class DatasetRootViolation(PipelineError):
    """Output path would write under a dataset root (hard invariant violation)."""


class MissingColumnError(PipelineError):
    """A required column is absent from a clinical CSV."""


class AdapterNotFound(PipelineError):
    pass
