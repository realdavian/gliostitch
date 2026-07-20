"""Pipeline-specific exceptions."""


class PipelineError(Exception):
    pass


class DatasetRootViolation(PipelineError):
    """Output path would write under a dataset root (hard invariant violation)."""


class MissingColumnError(PipelineError):
    """A required column is absent from a clinical CSV."""


class ClinicalKeyCollision(PipelineError):
    """Two clinical rows mapped to the same key, so one silently overwrote the other.

    Raised rather than tolerated: a collision means facts from one session are
    being attributed to another, which is invisible in the output.
    """


class AdapterNotFound(PipelineError):
    pass
