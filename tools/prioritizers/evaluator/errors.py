class InitializationError(Exception):
    pass


class TestDoesNotExistError(Exception):
    pass


class PrioritizationError(Exception):
    """Raised when the prioritization list is invalid."""
    pass
