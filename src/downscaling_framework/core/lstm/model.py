from dataclasses import dataclass, field
import numpy as np
from torch.utils.data import DataLoader
from torch import nn, optim
import torch


class ModelLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        pred_len: int,
        dropout_rate: float,
    ):
        super(ModelLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout_rate if num_layers > 1 else 0.0,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, pred_len)

    def forward(self, X):    
        
        if X.ndim == 2:
            X = X.unsqueeze(1)
            
        out, _ = self.lstm(X)
        # Pega a última saída da sequência temporal
        out = out[:, -1, :]
        return self.fc(out)


@dataclass
class LSTM:
    seq_len: int
    pred_len: int
    multivariate: bool

    input_size: int
    hidden_size: int
    num_layers: int
    dropout_rate: float
    lr: float
    device: str | None = None

    model: ModelLSTM = field(init=False)

    def __post_init__(self):
        if self.device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.model = ModelLSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            pred_len=self.pred_len,
            dropout_rate=self.dropout_rate,
        )
        self.model.to(device=self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    @classmethod
    def from_state(cls, state: dict) -> "LSTM":
        """Cria uma instância do modelo LSTM a partir do estado."""
        model = cls(
            seq_len=state["seq_len"],
            pred_len=state["pred_len"],
            multivariate=state["multivariate"],
            input_size=state["input_size"],
            hidden_size=state["hidden_size"],
            num_layers=state["num_layers"],
            dropout_rate=state["dropout_rate"],
            lr=state["lr"],
        )
        model.model.load_state_dict(state["state"])
        return model

    def execute_train(self, loader: DataLoader, epochs: int = 100) -> None:
        """Treina o modelo LSTM."""
        self.model.train()
        for epoch in range(epochs):
            for batch in loader:
                X_train, y_train = batch
                X_train = X_train.to(self.device, dtype=torch.float32)
                y_train = y_train.to(self.device, dtype=torch.float32)

                self.optimizer.zero_grad()
                outputs = self.model(X_train)
                loss = self.criterion(outputs, y_train)
                loss.backward()
                self.optimizer.step()

    def execute_predict(self, loader: DataLoader) -> np.ndarray:
        """Realiza previsões com o modelo LSTM."""
        self.model.eval()
        preds = []
        with torch.no_grad():
            for batch in loader:
                X_pred = batch[0].to(self.device, dtype=torch.float32)
                pred = self.model(X_pred)
                preds.append(pred.cpu().detach().numpy())

        return np.vstack(preds)

    def execute_evaluate(self, loader: DataLoader) -> float:
        """Avalia o modelo usando o erro quadrático médio."""
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in loader:
                X_valid, y_valid = batch
                X_valid = X_valid.to(self.device, dtype=torch.float32)
                y_valid = y_valid.to(self.device, dtype=torch.float32)

                outputs = self.model(X_valid)
                loss = self.criterion(outputs, y_valid)
                total_loss += loss.item()

        return total_loss / len(loader)

    def get_state(self) -> dict:
        """Retorna o estado do modelo."""
        return {
            "seq_len": self.seq_len,
            "pred_len": self.pred_len,
            "multivariate": self.multivariate,
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout_rate": self.dropout_rate,
            "lr": self.lr,
            "state": {k: v.cpu() for k, v in self.model.state_dict().items()},
        }