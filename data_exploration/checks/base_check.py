from abc import ABC, abstractmethod


class BaseCheck(ABC):

    def __init__(self, df):
        pass


    @abstractmethod
    def run_check(self):
        pass