from abc import ABC, abstractmethod
import numpy as np
from typing import Callable, Dict, Any


class BaseModel(ABC):
    """Базовый абстрактный класс для моделей машинного обучения.

    Этот класс определяет общий интерфейс для всех моделей в библиотеке ml_from_scratch.

        learning_rate: Скорость обучения для градиентного спуска. По умолчанию 0.01.
        l1: Коэффициент регуляризации L1 (Lasso). По умолчанию 0.0.
        l2: Коэффициент регуляризации L2 (Ridge). По умолчанию 0.0.
        verbose: Флаг для вывода информации о процессе обучения. По умолчанию False.
        loss_fn (Callable: Функция потерь для оптимизации.
        loss_history: Список значений функции потерь на каждой итерации.
    """
    def __init__(self, learning_rate: float = 0.01, l1: float = 0.0, l2: float = 0.0, verbose: bool = False):
        """
        Инициализирует базовую модель с заданными параметрами.
        """
        self.learning_rate = learning_rate
        self.l1 = l1
        self.l2 = l2
        self.verbose = verbose
        self.loss_fn: Callable[[np.ndarray, np.ndarray], float] = None
        self.loss_history: list = []

    @abstractmethod
    def fit(self, X:np.ndarray, y: np.ndarray) -> None:
        """
        Обучает модель на предоставленных данных.
        X: Матрица признаков формы.
        y: Вектор целевых значений формы.
        """

        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Предсказывает значения для заданных признаков.
        X: Матрица признаков
        np.ndarray: Вектор предсказаний
        """

        pass

    def set_loss(self, loss_fn: Callable[[np.ndarray, np.ndarray], float]) -> None:
        """
        Устанавливает пользовательскую функцию потерь
        loss_fn (Callable: Функция потерь, принимающая предсказания и истинные значения и возвращающая скаляр.
        """

        self.loss_fn = loss_fn
    
    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """
        Возвращает словарь с метриками и дополнительной информацией о модели.
        Dict[str, Any]: Словарь с метриками и данными
        """

        pass
