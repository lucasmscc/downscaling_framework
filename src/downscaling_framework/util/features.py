import copy
from dataclasses import dataclass
import torch

@dataclass
class Config:
    """Configurações usadas pelo modelo."""

    min_seq_len: int
    max_seq_len: int
    step: int
    lr_range: list[float]

    epochs: int
    n_trials: int
    patience: int    
    min_delta: float 

@dataclass
class EarlyStopping:
    """Interrompe o treinamento quando a validação deixa de melhorar."""

    patience: int
    min_delta: float
    best_loss: float = float("inf")
    counter: int = 0
    should_stop: bool = False
    best_state_dict: dict[str, torch.Tensor] | None = None

    def __call__(self, val_loss: float, model: torch.nn.Module) -> None:
        """Atualiza o estado do early stopping."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0

            self.best_state_dict = copy.deepcopy(model.state_dict())

        else:
            self.counter += 1

            if self.counter >= self.patience:
                self.should_stop = True
        pass

    def restore(self, model: torch.nn.Module) -> None:
        """Restaura os melhores pesos armazenados em memória."""
        if self.best_state_dict is None:
            raise RuntimeError("Nenhum estado do modelo foi armazenado.")

        model.load_state_dict(self.best_state_dict)