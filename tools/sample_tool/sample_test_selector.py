import random

from interfaces import TestSelector, SDCTest


class SampleTestSelector(TestSelector):
    def initialize(self, test_suite: list[SDCTest]) -> None:
        pass

    def select(self, test_suite: list[SDCTest]) -> list[bool]:
        return [random.random() < 0.5 for _ in test_suite]