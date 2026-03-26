from abc import ABC, abstractmethod
from typing import Optional
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np

class BaseModel(ABC):
    def __init__(self, model_name: str, base_preprocessor: Optional[Pipeline] = None) -> None:
        self.model_name = model_name
        self.base_preprocessor = base_preprocessor
        self.mdl = None         
        self.predictions = None

    @abstractmethod
    def train(self, data) -> None:
        """
        Train the model using ML Models for Multi-class and multi-label classification.
        """
        pass

    @abstractmethod
    def predict(self, X_test) -> None:
        pass

