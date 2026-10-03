from dataclasses import dataclass, field
import numpy as np
from torch.utils.data import DataLoader
from torch import nn, optim
import torch


class ModelCNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: list[int],
        pred_len: int,
        dropout_rate: float,
    ):
        super(ModelCNN, self).__init__()

        layers = []
        in_channels = input_size
        for hidden in hidden_size:
            layers.append(
                nn.Conv1d(
                    in_channels,
                    hidden,
                    kernel_size=3,
                    padding=1,
                )
            )
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden))
            layers.append(nn.Dropout(dropout_rate))
            in_channels = hidden

        self.cnn = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.heads = nn.ModuleList(
            [nn.Linear(hidden_size[-1], 1) for _ in range(pred_len)]
        )

    def forward(self, X):
        X = X.unsqueeze(-1)
        # X = X.permute(0, 2, 1)
        out = self.cnn(X)
        out = self.global_pool(out)
        out = out.squeeze(-1)  # (batch, hidden_size[-1])
        return torch.stack([h(out).squeeze(-1) for h in self.heads], dim=1)


@dataclass
class CNN:
    seq_len: int
    pred_len: int
    multivariate: bool

    input_size: int
    hidden_size: list[int]
    dropout_rate: float
    lr: float
    device: str | None = None

    model: ModelCNN = field(init=False)

    def __post_init__(self):
        if self.device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.model = ModelCNN(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            pred_len=self.pred_len,
            dropout_rate=self.dropout_rate,
        )
        self.model.to(device=self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    @classmethod
    def from_state(cls, state: dict) -> "CNN":
        """
        Cria uma instância do modelo a partir do estado.

        Args:
            state (dict): Estado do modelo.

        Returns:
            LSTM: Instância do modelo.
        """
        model = cls(
            seq_len=state["seq_len"],
            pred_len=state["pred_len"],
            multivariate=state["multivariate"],
            input_size=state["input_size"],
            hidden_size=state["hidden_size"],
            dropout_rate=state["dropout_rate"],
            lr=state["lr"],
        )
        model.model.load_state_dict(state["state"])
        return model

    def execute_train(self, loader: DataLoader, epochs: int = 100) -> None:
        """Treina o modelo.

        Args:
            loader (DataLoader): Dataloader com os dados de treinamento.
        """
        self.model.train()
        for epoch in range(epochs):
            for batch in loader:
                X_train, y_train = batch

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
            "dropout_rate": self.dropout_rate,
            "lr": self.lr,
            "state": {k: v.cpu() for k, v in self.model.state_dict().items()},
        }