import numpy as np
from typing import Dict, Any
from .base import BaseModel
import matplotlib.pyplot as plt
from io import BytesIO
import base64

class LinearRegression(BaseModel):
    """
    Класс для линейной регрессии с аналитическим и градиентным спуском.
    Поддерживает L1,L2 регуляризацию
    """
    
    def __init__(self, method:str = 'gradient', max_iter: int = 1000, tol: float = 1e-4, **kwargs):
        """
        Инициализация линейной регрессии 

        param method:Метод обучения (градиент или аналитический способ)
        param max_iter: Максимальное число итераций для градиентного спуска
        param tol:Переменная для поверки сходимости
        param ** kwarges: Доп. параметры(learning_rate,l1,l2,verbose)
        """
        if method not in ['analytic', 'gradient']:
            raise ValueError('Метод должен быть другой analytic или gradient')
        
        super().__init__(**kwargs)
        self.method = method
        self.max_iter = max_iter
        self.tol = tol
        self.weights: np.ndarray = None

        if self.loss_fn is None:
            self.loss_fn = lambda y_true, y_pred: np.mean((y_true - y_pred) ** 2) / 2
    
    def fit (self, X:np.ndarray, y:np.ndarray) -> None:
        """
        Обучение модели градиентным или аналитическим спуском

        param X: Матрица признаков
        param y: Вектор целевых переменных
        """

        if X.ndim != 2:
            raise ValueError ('X должен быть массивом!')
        if len(y.shape) == 1:
            y = y.reshape(-1,1)
        elif y.shape[1] != 1:
            raise ValueError('у должен быть вектором или столбцом')
        
        if X.shape[0] != y.shape[0]:
            raise ValueError('Количество образцов в X и y должно совпадать')
        
        # добавление столбца с 1
        n_samples, n_features = X.shape
        X_bias = np.c_[np.ones(n_samples), X]
        

        if self.method == 'analytic':
        # создаем матрицу для рег. L2
            n_params = X_bias.shape[1]
            matrix = self.l2 * np.eye(n_params)
            matrix[0, 0] = 0 

            # уравнения с l2 регуляризацией
            try:
                XTX = X_bias.T @ X_bias
                XTy = X_bias.T @ y
                self.weights = np.linalg.inv(XTX + matrix) @ XTy

                # вычисление предсказания
                y_pred = X_bias @ self.weights
                mse_loss = self.loss_fn(y, y_pred)
                l2_penalty = (self.l2 / 2) * np.sum(self.weights[1::] ** 2)
                total_loss = mse_loss + l2_penalty

                self.loss_history.append(float(total_loss))

                if self.verbose:
                    print(f'Аналитическое решение: final loss = {total_loss:6f}')
                       
            except np.linalg.LinAlgError as e:
                raise RuntimeError(f"Ошибка при вычислении весов: {e}")

        elif self.method == 'gradient':
            n_params = X_bias.shape[1]
            self.weights = np.random.randn(n_params, 1) * 0.01

            prev_loss = float('inf')
            for epoch in range(self.max_iter):
                y_pred = X_bias @ self.weights
                mse_loss = self.loss_fn(y, y_pred)
                l2_penalty = (self.l2 / 2) * np.sum(self.weights[1:] ** 2)
                l1_penalty = self.l1 * np.sum(np.abs(self.weights[1:]))
                total_loss = mse_loss + l2_penalty + l1_penalty
                self.loss_history.append(total_loss)
                
                if self.verbose and epoch % 100 == 0:
                    print(f'Epoch{epoch}: loss = {total_loss}:.6f')
                
                if abs(prev_loss - total_loss) < self.tol:
                    if self.verbose: 
                        print(f'Сошлись в одну эпоху {epoch}')
                    break
                prev_loss = total_loss

                gradient = (X_bias.T @ (y_pred - y)) / n_samples
                gradient[1:] += self.l2 * self.weights[1:]
                gradient[1:] += self.l1 * np.sign(self.weights[1:])
                self.weights -= self.learning_rate * gradient
            
            if self.verbose:
                print(f'итоговые потери за  {epoch} эпох: {total_loss:.6f}')

        else:
            raise ValueError("Неподдерживаемый метод")


    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Делает предсказания для переданных данных.

        param X: Матрица признаков.
        return: Вектор предсказанных значений.
        """
        if self.weights is None:
            raise ValueError('Модель не обучена. Сначала вызвать fit().')
        
        if X.ndim != 2:
            raise ValueError ('X должен быть двумерным массивом')
        
        n_samples = X.shape[0]
        X_bias = np.c_[np.ones(n_samples), X]

        y_pred = X_bias @ self.weights
        return y_pred.flatten()
    
    def set_loss(self, loss_fn: callable) -> None:
        """
        Устанавливает пользовательскую функцию потерь.

        param loss_fn: Лямбда-функция или функция потерь.
        """
        self.loss_fn = loss_fn

    def score(self, X: np.ndarray, y:np.ndarray) -> float:
        """
        Вычисляет R^2.

        param X: Матрица признаков.
        param y: Истинные значения.
        return: Значение R^2.
        """
        
        y_pred = self.predict(X)
        y = y.flatten()
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        if ss_tot == 0:
            return 0.0
        
        r2 = 1 - (ss_res / ss_tot)
        return float(r2)
    
    def get_additional_metrics(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Вычисляет дополнительные метрики (MAE, RMSE).

        param X: Матрица признаков.
        param y: Истинные значения.
        return: Словарь с метриками {'mae', 'rmse'}.
        """
        y_pred = self.predict(X)
        y = y.flatten()
        mae = np.mean(np.abs(y - y_pred))
        rmse = np.sqrt(np.mean(y - y_pred) ** 2)
        return {'mae': mae, 'rmse': rmse}
    
    @property
    def coef_(self):
        """sklearn-style: коэффициенты (без bias)"""
        return self.weights[1:].flatten() if self.weights is not None else None

    @property
    def intercept_(self):
        """sklearn-style: свободный член (bias)"""
        return float(self.weights[0]) if self.weights is not None else None
    
    def get_metrics(self) -> Dict[str, Any]:
        if not self.loss_history:
            return {'message': 'Модель не обучена'}

        metrics = {
            'final_loss': float(self.loss_history[-1]),
            'loss_history': self.loss_history.copy(),
            'coef_': self.coef_.tolist() if self.coef_ is not None else None,
            'intercept_': self.intercept_,  # ← УБРАЛИ [0]! Это float!
            'weights': self.weights.flatten().tolist()
        }

        # === График loss ===
        plt.figure(figsize=(8, 5))
        plt.plot(self.loss_history, 'o-' if len(self.loss_history) == 1 else '-', color='blue')
        plt.title('Training Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        metrics['loss_plot'] = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        plt.close()

        # === Если есть X_test и y_test ===
        if hasattr(self, 'X_test') and hasattr(self, 'y_test'):
            X_test = np.asarray(self.X_test)
            y_test = np.asarray(self.y_test).flatten()
            y_pred = self.predict(X_test).flatten()

            # MAE, RMSE
            add_metrics = self.get_additional_metrics(X_test, y_test)
            metrics.update(add_metrics)

            # R²
            ss_res = np.sum((y_test - y_pred) ** 2)
            ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
            metrics['r2'] = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

            # График регрессии
            idx = np.argsort(X_test.flatten())
            X_sorted = X_test[idx]
            y_pred_sorted = y_pred[idx]

            plt.figure(figsize=(10, 6))
            plt.scatter(X_test, y_test, color='lightblue', label='Тест', alpha=0.7, s=60)
            plt.plot(X_sorted, y_pred_sorted, color='red', linewidth=2, label='Предсказание')
            plt.title('Линейная регрессия')
            plt.xlabel('X')
            plt.ylabel('y')
            plt.legend()
            plt.grid(True, alpha=0.3)
            buf2 = BytesIO()
            plt.savefig(buf2, format='png', bbox_inches='tight', dpi=100)
            buf2.seek(0)
            metrics['regression_plot'] = base64.b64encode(buf2.read()).decode('utf-8')
            buf2.close()
            plt.close()

        return metrics