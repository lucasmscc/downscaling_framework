import numpy as np
from loguru import logger

from downscaling_framework.core.elasticnet.model import ElasticNet


def execute(
    X_test: np.ndarray,
    model_state: dict,
) -> np.ndarray:
    """Executa a previsão do modelo ElasticNet com base em vetores X e estado salvo."""
    logger.info("Previsão do modelo ElasticNet iniciada.")

    model = ElasticNet.from_state(model_state)
    preds = model.execute_predict(X_test)

    return preds