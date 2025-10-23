from abc import ABC, abstractmethod
import numpy as np
from typing import Callable, Dict, Any


class BaseModel(ABC):
    def __init__(self, learning_rate: float = 0.01, l1: float = 0.0, l2: float = 0.0, verbose: bool = False):
        self.learning_rate = learning_rate
        self.l1 = l1
        self.l2 = l2
        self.verbose = verbose
        self.loss_fn: Callable[[np.ndarray, np.ndarray], float] = None
        self.loss_history: list = []

    @abstractmethod
    def fit(self, X:np.ndarray, y: np.ndarray) -> None:

        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:

        pass

    def set_loss(self, loss_fn: Callable[[np.ndarray, np.ndarray], float]) -> None:

        self.loss_fn = loss_fn
    
    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:

        pass
