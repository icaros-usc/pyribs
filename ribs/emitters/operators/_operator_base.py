"""Operator Base Class"""
from abc import ABC, abstractmethod


class OperatorBase(ABC):
    """Abstract Interface"""

    @abstractmethod
    def __init__(self):
        """Pass"""

    @abstractmethod
    def operate(self):
        """Pass"""
