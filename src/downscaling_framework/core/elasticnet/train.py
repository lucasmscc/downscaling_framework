import optuna
import numpy as np
from loguru import logger

from downscaling_framework.core.elasticnet.model import ElasticNet
from downscaling_framework.util.features import Config


def execute(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    configs: dict,
) -> ElasticNet:
    """Executa o treinamento e a otimização de hiperparâmetros do ElasticNet via Optuna."""
    logger.info("Treinamento do modelo ElasticNet iniciado.")
    
    config = Config(**configs)

    def objective(trial: optuna.Trial) -> float:
        alpha = trial.suggest_float("alpha", 1e-5, 1e1, log=True)
        l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)

        pred_len = y_train.shape[1] if y_train.ndim > 1 else 1

        model = ElasticNet(
            seq_len=1,
            pred_len=pred_len,
            multivariate=True,
            alpha=alpha,
            l1_ratio=l1_ratio,
        )

        model.execute_train(X_train, y_train)
        val_loss = model.execute_evaluate(X_valid, y_valid)
        
        trial.set_user_attr("model", model)
        return val_loss

    study = optuna.create_study(study_name="ElasticNet", direction="minimize")
    study.optimize(objective, n_trials=config.n_trials)
    logger.info(f"Melhor trial encontrado (MSE): {study.best_value}")

    best_model = study.best_trial.user_attrs["model"]
    return best_model