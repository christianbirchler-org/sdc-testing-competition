import random

from interfaces import TestSelector, SDCTest


class SampleTestSelector(TestSelector):
    def select(self, test_suite: list[SDCTest]) -> list[bool]:
        return [random.random() < 0.5 for _ in test_suite]