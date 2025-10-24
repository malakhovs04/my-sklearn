import numpy as np
from typing import Dict, Any, Optional
from .base import BaseModel
import matplotlib.pyplot as plt
from io import BytesIO
import base64


class LogisticRegression(BaseModel):
    """
    Логистическая регрессия с градиентным спуском.
    Поддерживает L1 и L2 регуляризацию.
    """

    def __init__(self, max_iter: int = 1000, tol: float = 1e-4, **kwargs):
        super().__init__(**kwargs)
        self.max_iter = max_iter
        self.tol = tol
        self.weights: Optional[np.ndarray] = None

        if self.loss_fn is None:
            self.loss_fn = lambda y_true, y_pred: -np.mean(
                y_true * np.log(y_pred + 1e-15) + (1 - y_true) * np.log(1 - y_pred + 1e-15)
            )

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Сигмоида с защитой от переполнения."""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LogisticRegression':
        """Обучает модель."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1, 1)

        if X.ndim != 2 or y.shape[0] != X.shape[0] or not np.all(np.isin(y, [0, 1])):
            raise ValueError("Некорректные данные")

        n_samples, n_features = X.shape
        X_bias = np.c_[np.ones(n_samples), X]
        self.weights = np.random.randn(n_features + 1, 1) * 0.01

        prev_loss = float('inf')
        for epoch in range(self.max_iter):
            z = X_bias @ self.weights
            y_pred = self._sigmoid(z)

            loss = self.loss_fn(y, y_pred)
            l2_reg = (self.l2 / 2) * np.sum(self.weights[1:] ** 2)
            l1_reg = self.l1 * np.sum(np.abs(self.weights[1:]))
            total_loss = loss + l2_reg + l1_reg
            self.loss_history.append(float(total_loss))

            if abs(prev_loss - total_loss) < self.tol:
                if self.verbose:
                    print(f"Сходимость на эпохе {epoch}")
                break
            prev_loss = total_loss

            gradient = (X_bias.T @ (y_pred - y)) / n_samples
            gradient[1:] += self.l2 * self.weights[1:]
            if self.l1 > 0:
                gradient[1:] += self.l1 * np.sign(self.weights[1:])

            self.weights -= self.learning_rate * gradient

            if self.verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}: loss = {total_loss:.6f}")

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Возвращает [[P(y=0), P(y=1)], ...]"""
        if self.weights is None:
            raise ValueError("Модель не обучена")
        X = np.asarray(X, dtype=float)
        X_bias = np.c_[np.ones(X.shape[0]), X]
        proba_1 = self._sigmoid(X_bias @ self.weights).flatten()
        return np.column_stack([1 - proba_1, proba_1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Предсказывает классы."""
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Accuracy."""
        y_pred = self.predict(X)
        y_true = np.asarray(y).flatten()
        return float(np.mean(y_pred == y_true))

    def get_additional_metrics(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Precision, Recall, F1, ROC-AUC """
        y_pred = self.predict(X).flatten()
        y_true = np.asarray(y).flatten()
        y_proba = self.predict_proba(X)[:, 1]

        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        desc_idx = np.argsort(y_proba)[::-1]
        y_true_s = y_true[desc_idx]
        y_proba_s = y_proba[desc_idx]

        thresholds = np.unique(y_proba_s)[::-1]
        tpr_list = [0.0]
        fpr_list = [0.0]

        for thresh in thresholds:
            pred = (y_proba_s >= thresh).astype(int)
            tp_t = np.sum((pred == 1) & (y_true_s == 1))
            fp_t = np.sum((pred == 1) & (y_true_s == 0))
            fn_t = np.sum((pred == 0) & (y_true_s == 1))
            tn_t = np.sum((pred == 0) & (y_true_s == 0))

            tpr = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0.0
            fpr = fp_t / (fp_t + tn_t) if (fp_t + tn_t) > 0 else 0.0
            tpr_list.append(tpr)
            fpr_list.append(fpr)

        tpr_list.append(1.0)
        fpr_list.append(1.0)

        auc_score = 0.0
        for i in range(1, len(fpr_list)):
            auc_score += (fpr_list[i] - fpr_list[i-1]) * (tpr_list[i] + tpr_list[i-1]) / 2

        return {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'roc_auc': float(auc_score)
        }

    @property
    def coef_(self):
        return self.weights[1:].flatten() if self.weights is not None else None

    @property
    def intercept_(self):
        return float(self.weights[0]) if self.weights is not None else None

    def get_metrics(self) -> Dict[str, Any]:
        """Возвращает метрики + графики в base64."""
        if not self.loss_history:
            return {"message": "Модель не обучена"}

        metrics = {
            'final_loss': float(self.loss_history[-1]),
            'loss_history': self.loss_history.copy(),
            'coef_': self.coef_.tolist() if self.coef_ is not None else None,
            'intercept_': self.intercept_,
            'weights': self.weights.flatten().tolist()
        }

        plt.figure(figsize=(8, 5))
        plt.plot(self.loss_history, color='blue')
        plt.title('Training Loss'); plt.xlabel('Iteration'); plt.ylabel('Loss'); plt.grid(True, alpha=0.3)
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        metrics['loss_plot'] = base64.b64encode(buf.read()).decode('utf-8')
        buf.close(); plt.close()

        if hasattr(self, 'X_test') and hasattr(self, 'y_test'):
            add = self.get_additional_metrics(self.X_test, self.y_test)
            metrics.update(add)
            metrics['accuracy'] = self.score(self.X_test, self.y_test)

            y_pred = self.predict(self.X_test)
            tp = np.sum((y_pred == 1) & (self.y_test == 1))
            fp = np.sum((y_pred == 1) & (self.y_test == 0))
            fn = np.sum((y_pred == 0) & (self.y_test == 1))
            tn = np.sum((y_pred == 0) & (self.y_test == 0))
            cm = np.array([[tn, fp], [fn, tp]])
            metrics['confusion_matrix'] = cm.tolist()

            plt.figure(figsize=(6, 5))
            plt.imshow(cm, cmap='Blues')
            plt.title('Confusion Matrix'); plt.colorbar()
            plt.xticks([0,1]); plt.yticks([0,1])
            plt.xlabel('Predicted'); plt.ylabel('True')
            for i in range(2):
                for j in range(2):
                    plt.text(j, i, str(cm[i][j]), ha='center', va='center')
            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            metrics['cm_plot'] = base64.b64encode(buf.read()).decode('utf-8')
            buf.close(); plt.close()

            y_proba = self.predict_proba(self.X_test)[:, 1]
            desc_idx = np.argsort(y_proba)[::-1]
            y_true_s = np.asarray(self.y_test).flatten()[desc_idx]
            y_proba_s = y_proba[desc_idx]

            thresholds = np.unique(y_proba_s)[::-1]
            tpr_list = [0.0]
            fpr_list = [0.0]
            for thresh in thresholds:
                pred = (y_proba_s >= thresh).astype(int)
                tp_t = np.sum((pred == 1) & (y_true_s == 1))
                fp_t = np.sum((pred == 1) & (y_true_s == 0))
                fn_t = np.sum((pred == 0) & (y_true_s == 1))
                tn_t = np.sum((pred == 0) & (y_true_s == 0))
                tpr = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0.0
                fpr = fp_t / (fp_t + tn_t) if (fp_t + tn_t) > 0 else 0.0
                tpr_list.append(tpr)
                fpr_list.append(fpr)
            tpr_list.append(1.0)
            fpr_list.append(1.0)

            plt.figure(figsize=(8, 6))
            plt.plot(fpr_list, tpr_list, color='darkorange', lw=2, label=f'AUC = {add["roc_auc"]:.3f}')
            plt.plot([0,1],[0,1],'k--')
            plt.xlim([0,1]); plt.ylim([0,1.05])
            plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC Curve'); plt.legend(); plt.grid(True, alpha=0.3)
            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            metrics['roc_plot'] = base64.b64encode(buf.read()).decode('utf-8')
            buf.close(); plt.close()

        return metrics