from abc import ABC, abstractmethod

class ClassifierInterface(ABC):
    @abstractmethod
    def fit(self, *args, **kwargs):
        pass

    @abstractmethod
    def my_predict(self, *args, **kwargs):
        pass

    @abstractmethod
    def validate(self, *args, **kwargs):
        pass