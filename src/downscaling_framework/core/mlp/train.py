import optuna
import numpy as np
import pandas as pd
import torch
from loguru import logger
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from downscaling_framework.core.mlp.model import MLP  # Substitua pelo caminho correto do seu módulo
from downscaling_framework.util.features import Config, EarlyStopping  # Mantenha os utilitários genéricos se existirem


def execute(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    configs: dict,
) -> MLP:
    """
    Executa o treinamento e a otimização de hiperparâmetros do modelo MLP usando vetores X e y.

    Args:
        X_train (np.ndarray): Dados de entrada de treino.
        y_train (np.ndarray): Alvo de treino.
        X_valid (np.ndarray): Dados de entrada de validação.
        y_valid (np.ndarray): Alvo de validação.
        scaler_X (StandardScaler): Scaler para as features.
        scaler_y (StandardScaler): Scaler para o alvo.
        configs (dict): Dicionário de configuração para o Optuna/Modelo.

    Returns:
        MLP: Instância do melhor modelo treinado.
    """
    logger.info("Treinamento do modelo MLP iniciado.")

    config = Config(**configs)

    def objective(trial: optuna.Trial) -> float:
        # Hiperparâmetros otimizados via Optuna
        n_layers = trial.suggest_int("n_layers", 1, 4)
        hidden_size = []
        for layer in range(n_layers):
            neurons = 2 ** trial.suggest_int(f"layer_{layer + 1}", 3, 8)
            hidden_size.append(neurons)

        activation = trial.suggest_categorical(
            "activation",
            ["relu", "tanh", "sigmoid"],
        )
        dropout = trial.suggest_float("dropout", 0.1, 0.5, step=0.1)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

        # Configurações dimensionais do modelo
        input_size = X_train.shape[1]
        pred_len = y_train.shape[1] if y_train.ndim > 1 else 1

        model = MLP(
            seq_len=1,  # Como X já é tabular/vetor, tratamos a sequência como 1 passo ou ignoramos no construtor se ajustado
            pred_len=pred_len,
            multivariate=True,
            input_size=input_size,
            hidden_size=hidden_size,
            activation=activation,
            dropout_rate=dropout,
            lr=lr,
        )

        # Criação dos Datasets com TensorDataset
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
            patience=5,
            min_delta=1e-5,
        )

        try:
            for epoch in range(config.epochs):
                # Executa 1 época por vez para avaliar e aplicar o Early Stopping
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

    study = optuna.create_study(study_name="MLP", direction="minimize")
    study.optimize(objective, n_trials=config.n_trials)
    logger.info(f"Melhor trial encontrado (MSE): {study.best_value}")

    best_model = study.best_trial.user_attrs["model"]
    return best_model