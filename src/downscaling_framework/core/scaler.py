from dataclasses import dataclass, field

import numpy as np
from sklearn.preprocessing import StandardScaler

@dataclass
class Scaler:
    
    scaler_x_mean: np.ndarray = field(default_factory=lambda: np.array([]))
    scaler_x_scaler: np.ndarray = field(default_factory=lambda: np.array([]))
    scaler_y_mean: float = 0.0
    scaler_y_scaler: float = 0.0

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        # Aplicando normalização
        scaler_X = StandardScaler().fit(X)
        scaler_y = StandardScaler().fit(y)

        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y)

        self.scaler_x_mean = scaler_X.mean_
        self.scaler_x_scaler = scaler_X.scale_
        self.scaler_y_mean = scaler_y.mean_
        self.scaler_y_scaler = scaler_y.scale_

        return X_scaled, y_scaled

    def forecast(
        self,
        X: np.ndarray | None = None,
        y: np.ndarray | None = None,
    ) -> np.ndarray:
        # Aplicando normalização
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        scaler_X.mean_ = self.scaler_x_mean
        scaler_X.scale_ = self.scaler_x_scaler
        scaler_y.mean_ = self.scaler_y_mean
        scaler_y.scale_ = self.scaler_y_scaler

        if X is not None and y is not None:
            X_scaled = scaler_X.transform(X)
            y_scaled = scaler_y.transform(y)
            return X_scaled, y_scaled
        elif X is not None:
            return scaler_X.transform(X)
        elif y is not None:
            return scaler_y.transform(y)

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        # Aplicando normalização
        scaler_y = StandardScaler()
        scaler_y.mean_ = self.scaler_y_mean
        scaler_y.scale_ = self.scaler_y_scaler

        y_rescaled = scaler_y.inverse_transform(y)
        return y_rescaled

    def get_state(self) -> dict:
        return {
            "scaler_x_mean": self.scaler_x_mean,
            "scaler_x_scaler": self.scaler_x_scaler,
            "scaler_y_mean": self.scaler_y_mean,
            "scaler_y_scaler": self.scaler_y_scaler,
        }

    @classmethod
    def from_state(cls, state: dict) -> "Scaler":
        return cls(
            scaler_x_mean=state["scaler_x_mean"],
            scaler_x_scaler=state["scaler_x_scaler"],
            scaler_y_mean=state["scaler_y_mean"],
            scaler_y_scaler=state["scaler_y_scaler"],
        )