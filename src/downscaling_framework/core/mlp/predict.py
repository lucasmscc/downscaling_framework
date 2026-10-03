import numpy as np
import torch
from loguru import logger
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from downscaling_framework.core.mlp.model import MLP  # Substitua pelo caminho correto do seu módulo


def execute(
    X_test: np.ndarray,
    model_state: dict,
) -> np.ndarray:
    """
    Executa a previsão do modelo MLP com base em vetores X e estado salvo.

    Args:
        X_test (np.ndarray): Dados de entrada de teste.
        scaler_X (StandardScaler): Scaler utilizado nas features.
        scaler_y (StandardScaler): Scaler utilizado no alvo (para inversão da escala).
        model_state (dict): Dicionário contendo o estado do modelo.

    Returns:
        np.ndarray: Previsões geradas e com a escala original revertida.
    """
    logger.info("Previsão do modelo MLP iniciada.")

    # Carrega o modelo a partir do estado salvo
    model = MLP.from_state(model_state)

    # Criação do Dataset e DataLoader para inferência
    # Passamos um dummy target apenas para manter a compatibilidade do TensorDataset se necessário, 
    # ou criamos o DataLoader apenas com X caso o execute_predict aceite. 
    # Olhando o seu código original de predict, ele consome batch[0], então podemos passar X e zeros.
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