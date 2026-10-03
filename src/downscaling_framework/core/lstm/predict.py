import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader, TensorDataset

from downscaling_framework.core.lstm.model import LSTM


def execute(
    X_test: np.ndarray,
    model_state: dict,
) -> np.ndarray:
    """Executa a previsão do modelo LSTM com base em vetores X e estado salvo."""
    logger.info("Previsão do modelo LSTM iniciada.")

    model = LSTM.from_state(model_state)

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

    preds = model.execute_predict(test_loader)
    return preds