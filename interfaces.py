import abc


class SDCTest:
    def __init__(self):
        pass


class TestSelector(abc.ABC):
    @abc.abstractmethod
    def initialize(self, test_suite: list[SDCTest]) -> None:
        """
        Initialize the test selector with an existing test suite so that implementations of
        this method can train machine learning models if needed.

        :param test_suite:
        :return:
        """

    @abc.abstractmethod
    def select(self, test_suite: list[SDCTest]) -> list[bool]:
        """

        :param test_suite:
        :return: List of booleans
        """
        pass

    @abc.abstractmethod
    def name(self) -> str:
        """

        :return: The name of the test selector
        """
        pass
