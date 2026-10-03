from dataclasses import dataclass, field

import numpy as np
from torch import nn, optim
import torch
from torch.utils.data import DataLoader


class ModelMLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: list[int],
        n_horizons: int,
        activation: str,
        dropout_rate: float,
    ):
        super(ModelMLP, self).__init__()

        activation_map = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
        }
        activation_cls = activation_map[activation]

        layers = []
        prev_size = input_size
        for hidden in hidden_size:
            layers.append(nn.Linear(prev_size, hidden))
            layers.append(activation_cls())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden

        self.backbone = nn.Sequential(*layers)
        self.heads = nn.ModuleList([nn.Linear(prev_size, 1) for _ in range(n_horizons)])

    def forward(self, X):
        features = self.backbone(X)
        return torch.stack([h(features).squeeze(-1) for h in self.heads], dim=1)


@dataclass
class MLP:
    seq_len: int
    pred_len: int
    multivariate: bool

    input_size: int
    hidden_size: list[int]
    activation: str
    dropout_rate: float
    lr: float
    device: str | None = None

    model: ModelMLP = field(init=False)

    def __post_init__(self):
        if self.device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.model = ModelMLP(
            self.input_size,
            self.hidden_size,
            self.pred_len,
            self.activation,
            self.dropout_rate,
        )
        self.model.to(device=self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    @classmethod
    def from_state(cls, state: dict) -> "MLP":
        """
        Cria uma instância do modelo a partir do estado.

        Args:
            state (dict): Estado do modelo.

        Returns:
            MLP: Instância do modelo.
        """
        model = cls(
            seq_len=state["seq_len"],
            pred_len=state["pred_len"],
            multivariate=state["multivariate"],
            input_size=state["input_size"],
            hidden_size=state["hidden_size"],
            activation=state["activation"],
            dropout_rate=state["dropout_rate"],
            lr=state["lr"],
        )
        model.model.load_state_dict(state["state"])
        return model

    def execute_train(self, loader: DataLoader, epochs: int = 100) -> None:
        """Treina o modelo.

        Args:
            loader (DataLoader): Dataloader com os dados de treinamento.
            epochs (int, optional): Número de épocas para treinamento. Defaults to 100.
        """
        self.model.train()
        for _ in range(epochs):
            for batch in loader:
                X_train, y_train = batch
                X_train = X_train.reshape(X_train.shape[0], -1)

                # Verifica se o dispositivo é CUDA e move os dados para lá
                X_train = X_train.to(self.device, dtype=torch.float32)
                y_train = y_train.to(self.device, dtype=torch.float32)

                self.optimizer.zero_grad()
                outputs = self.model(X_train)
                loss = self.criterion(outputs, y_train)
                loss.backward()
                self.optimizer.step()

    def execute_predict(self, loader: DataLoader) -> np.ndarray:
        """Realiza previsões com o modelo.

        Args:
            loader (DataLoader): Dataloader com os dados de teste.

        Returns:
            ndarray: Previsões do modelo.
        """
        self.model.eval()
        preds = []
        with torch.no_grad():
            for batch in loader:
                X_pred = batch[0]
                X_pred = X_pred.reshape(X_pred.shape[0], -1)

                # Verifica se o dispositivo é CUDA e move os dados para lá
                X_pred = X_pred.to(self.device, dtype=torch.float32)

                pred = self.model(X_pred)
                preds.append(pred.cpu().detach().numpy())

        return np.vstack(preds)

    def execute_evaluate(self, loader: DataLoader) -> float:
        """Avalia o modelo usando o erro quadrático médio.

        Args:
            loader (DataLoader): Dataloader com os dados de validação.

        Returns:
            float: Erro quadrático médio do modelo.
        """
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in loader:
                X_valid, y_valid = batch
                X_valid = X_valid.reshape(X_valid.shape[0], -1)

                # Verifica se o dispositivo é CUDA e move os dados para lá
                X_valid = X_valid.to(self.device, dtype=torch.float32)
                y_valid = y_valid.to(self.device, dtype=torch.float32)

                outputs = self.model(X_valid)
                loss = self.criterion(outputs, y_valid)
                total_loss += loss.item()

        return total_loss / len(loader)

    def get_state(self) -> dict:
        """
        Retorna o estado do modelo.

        Returns:
            dict: Estado do modelo.
        """
        return {
            "seq_len": self.seq_len,
            "pred_len": self.pred_len,
            "multivariate": self.multivariate,
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "activation": self.activation,
            "dropout_rate": self.dropout_rate,
            "lr": self.lr,
            "state": {k: v.cpu() for k, v in self.model.state_dict().items()},
        }