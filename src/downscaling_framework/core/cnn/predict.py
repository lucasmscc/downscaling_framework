import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader, TensorDataset

from downscaling_framework.core.cnn.model import CNN


def execute(
    X_test: np.ndarray,
    model_state: dict,
) -> np.ndarray:
    """
    Executa a previsão do modelo CNN com base em vetores X e estado salvo.

    Args:
        X_test (np.ndarray): Dados de entrada de teste.
        scaler_X (StandardScaler): Scaler utilizado nas features.
        scaler_y (StandardScaler): Scaler utilizado no alvo (para inversão da escala).
        model_state (dict): Dicionário contendo o estado do modelo.

    Returns:
        np.ndarray: Previsões geradas e com a escala original revertida.
    """
    logger.info("Previsão do modelo CNN iniciada.")


    # Carrega o modelo a partir do estado salvo
    model = CNN.from_state(model_state)

    # Criação do TensorDataset para inferência com um alvo fictício para compatibilidade
    dummy_y = np.zeros((X_test.shape[0], model.pred_len))
    ds_test = TensorDataset(
        torch.tensor(X_test).float(),
        torch.tensor(dummy_y).float()
    )
    
    test_loader = DataLoader(
        ds_test,
        batch_size=128,
        shuffle=False,
    )

    # Executa a predição
    preds = model.execute_predict(test_loader)

    return preds