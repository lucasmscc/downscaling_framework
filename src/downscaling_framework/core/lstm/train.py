import optuna
import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader, TensorDataset

from downscaling_framework.core.lstm.model import LSTM
from downscaling_framework.util.features import Config, EarlyStopping


def execute(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    configs: dict,
) -> LSTM:
    """Executa o treinamento e a otimização de hiperparâmetros do modelo LSTM usando Optuna."""
    logger.info("Treinamento do modelo LSTM iniciado.")
    
    config = Config(**configs)

    def objective(trial: optuna.Trial) -> float:
        hidden_size = 2 ** trial.suggest_int("hidden_size_exp", 4, 8) # 16 a 256
        num_layers = trial.suggest_int("num_layers", 1, 3)
        dropout = trial.suggest_float("dropout", 0.1, 0.5, step=0.1) if num_layers > 1 else 0.0
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

        input_size = X_train.shape[1] if X_train.ndim > 1 else 1
        pred_len = y_train.shape[1] if y_train.ndim > 1 else 1

        model = LSTM(
            seq_len=1,
            pred_len=pred_len,
            multivariate=True,
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout_rate=dropout,
            lr=lr,
        )

        ds_train = TensorDataset(
            torch.tensor(X_train).float(),
            torch.tensor(y_train).float()
        )
        ds_valid = TensorDataset(
            torch.tensor(X_valid).float(),
            torch.tensor(y_valid).float()
        )

        train_loader = DataLoader(
            ds_train,
            batch_size=config.batch_size if hasattr(config, "batch_size") else 64,
            shuffle=True,
        )
        valid_loader = DataLoader(
            ds_valid,
            batch_size=config.batch_size if hasattr(config, "batch_size") else 64,
            shuffle=False,
        )

        early_stopping = EarlyStopping(
            patience=config.patience,
            min_delta=config.min_delta,
        )

        try:
            for epoch in range(config.epochs):
                model.execute_train(train_loader, epochs=1)
                val_loss = model.execute_evaluate(valid_loader)

                logger.debug(
                    f"Trial {trial.number} | Epoch {epoch + 1}/{config.epochs} "
                    f"- val_loss: {val_loss:.6f}"
                )

                early_stopping(val_loss, model.model)
                if early_stopping.should_stop:
                    logger.debug(f"Early stopping acionado na época {epoch + 1}")
                    break

        except torch.cuda.OutOfMemoryError:
            logger.warning("Erro de Memória no CUDA durante o Optuna.")
            return float("inf")

        early_stopping.restore(model.model)
        trial.set_user_attr("model", model)

        val_loss = model.execute_evaluate(valid_loader)
        return val_loss

    study = optuna.create_study(study_name="LSTM", direction="minimize")
    study.optimize(objective, n_trials=config.n_trials)
    logger.info(f"Melhor trial encontrado (MSE): {study.best_value}")

    best_model = study.best_trial.user_attrs["model"]
    return best_model