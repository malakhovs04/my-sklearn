import numpy as np
from typing import Dict, Any
from .base import BaseModel
import matplotlib.pyplot as plt
from io import BytesIO
import base64

class Logisticregression(BaseModel):
    
    def __init__(self, max_iter: int = 1000, tol: float = 1e-4, **kwargs):
        super().__init__(**kwargs)
        self.max_iter = max_iter
        self.tol = tol
        self.weights: np.ndarray = None

        if self.loss_fn is None:
            self.loss_fn = lambda y_true, y_pred: -np.mean(y_true * np.log(y_pred + 1e-15) + (1 - y_true) * np.log(1 - y_pred + 1e-15))

    def _sigmoid_(self, z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def fit (self, X:np.ndarray, y:np.ndarray) -> None:

        if X.ndim != 2:
            raise ValueError ('X должен быть двухмерным массивом')
        if y.shape[0] != X.shape[0]:
            raise ValueError ('Количество образцов должно совпадать')
        if not np.all(np.isin(y, [0,1])):
            raise ValueError(' y должен содержать только 0 и 1')
        
        y = y.reshape(-1, 1)
        n_samples, n_features = X.shape
        X_bias = np.c_[np.ones(n_samples), X]

        self.weights = np.random.randn(n_features + 1, 1) * 0,01

        prev_loss = float('int')
        for epoch in range(self.max_iter):
            z = X_bias @ self.weights
            y_pred = self._sigmoid_(z)
            loss = self.loss_fn(y, y_pred)
            l2_penalty = (self.l2 / 2) * np.sum(self.weights[1:] ** 2)
            l1_penalty = self.l1 * np.sum(np.abs(self.weights[1:]))
            total_loss = loss + l2_penalty + l1_penalty
            self.loss_history.append(total_loss)

            if self.verbose and epoch % 100 == 0:
                print(f'Epoch {epoch}: loss = {total_loss:.6f}')
            
            if abs(prev_loss - total_loss) < self.tol:
                if self.verbose:
                    print(f'Сошлись в одну эпоху {epoch}')
                break
            prev_loss = total_loss

            gradient = (X_bias.T @ (y_pred - y)) / n_samples
            gradient[1:] += self.l2 * self.weights[1:]
            gradient[1:] -= self.learning_rate * gradient

        if self.verbose:
            print(f'Окончательная потеря {epoch} эпохи {total_loss:.6f}')

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise ValueError('Модель не обучена')
        if X.ndim != 2:
            raise ValueError('X долженн быть двумерным массивом')
        n_samples = X.shape[0]
        X_bias = np.c_[np.ones(n_samples), X]
        y_pred_prob = self._sigmoid_(X_bias @ self.weights)
        return(y_pred_prob >= 0.5).astype(int).flatten()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise ValueError('Модель не обучена')
        n_samples = X.shape[0]
        X_bias = np.c_[np.ones(n_samples), X]
        return self._sigmoid(X_bias @ self.weights).flatten()
    
    def score (self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
    
    def get_additional_metrics(self, X:np.ndarray, y:np.ndarray) -> Dict[str, float]:
        y_pred = self.predict(X)
        y_prob = self.predict_proba(X)
        tp = np.sum((y_pred == 1) & (y == 1))
        fp = np.sum((y_pred == 1) & (y == 0))
        fn = np.sum((y_pred == 1) & (y == 0))
    
    
    def get_metrics(self) -> dict[str, Any]:
        if not self.loss_history:
            return {"message": "Модель не обучена"}

        metrics = {
            "final_loss": self.loss_history[-1],
            "loss_history": self.loss_history,
            "weights": self.weights.flatten().tolist() if self.weights is not None else None
        }

        plt.figure(figsize=(10, 6))
        plt.plot(self.loss_history, label="Loss History")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("Training Loss Over Iterations")
        plt.legend()
        plt.grid(True)

        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        loss_plot_base64 = base64.b64encode(buf.read()).decode("utf-8")
        buf.close()
        plt.close()

        metrics["loss_plot"] = loss_plot_base64

        return metrics