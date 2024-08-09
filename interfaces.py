import abc


class SDCTest:
    def __init__(self):
        pass


class TestSelector(abc.ABC):
    @abc.abstractmethod
    def select(self, test_suite: list[SDCTest]) -> list[bool]:
        """

        :rtype: List of booleans
        :param test_suite:
        """
        pass
