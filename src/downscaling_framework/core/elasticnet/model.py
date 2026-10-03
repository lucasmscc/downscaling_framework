from dataclasses import dataclass, field
import numpy as np
from sklearn.linear_model import ElasticNet as SklearnElasticNet


@dataclass
class ElasticNet:
    seq_len: int
    pred_len: int
    multivariate: bool
    alpha: float
    l1_ratio: float
    random_state: int = 42

    model: SklearnElasticNet = field(init=False)

    def __post_init__(self):
        self.model = SklearnElasticNet(
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            random_state=self.random_state
        )

    @classmethod
    def from_state(cls, state: dict) -> "ElasticNet":
        """Cria uma instância do modelo a partir do estado salvo."""
        model = cls(
            seq_len=state["seq_len"],
            pred_len=state["pred_len"],
            multivariate=state["multivariate"],
            alpha=state["alpha"],
            l1_ratio=state["l1_ratio"],
            random_state=state.get("random_state", 42),
        )
        model.model = state["state"]
        return model

    def execute_train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Treina o modelo ElasticNet."""
        self.model.fit(X, y)

    def execute_predict(self, X: np.ndarray) -> np.ndarray:
        """Realiza previsões com o modelo."""
        return self.model.predict(X)

    def execute_evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """Avalia o modelo usando o erro quadrático médio (MSE)."""
        preds = self.execute_predict(X)
        return float(np.mean((preds - y) ** 2))

    def get_state(self) -> dict:
        """Retorna o estado do modelo."""
        return {
            "seq_len": self.seq_len,
            "pred_len": self.pred_len,
            "multivariate": self.multivariate,
            "alpha": self.alpha,
            "l1_ratio": self.l1_ratio,
            "random_state": self.random_state,
            "state": self.model,
        }