import importlib
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split

from downscaling_framework.core.scaler import Scaler
from downscaling_framework.gribs_reader import process_gribs
from downscaling_framework.util.config import CONFIG
from downscaling_framework.util.model_tools import extract_grid_matrix


def execute(
    states: dict,
    target_variable: str,
    release_range: tuple[datetime],
    latitude: float,
    longitude: float,
    config: dict,
    nwp_path: Path,
) -> None:
    """
    Executa o treinamento.

    Args:
        target_variable (str):
            Variável alvo para previsão.
        release_range  (tuple[datetime]):
            Timestamp de previsão do modelo.
        latitude (float):
            Latitude do anemometro ou parque eólico de interesse.
        longitude (float):
            Longitude do anemometro ou parque eólico de interesse.
        config (dict):
            Dicionário de configuração de leitura dos gribs.
        nwp_path (Path):
            Diretório com os arquivos do GFS.

    """

    models = list(CONFIG.models.keys())

    if len(release_range) == 2:
        start_date = release_range[0]
        end_date = release_range[1]
    else:
        start_date = release_range[0] - pd.DateOffset(months=6)
        end_date = release_range[0]

    nwp = process_gribs(
        configs=config,
        root_directory=nwp_path,
        coordinates=[latitude,longitude + 360],
        end_date=end_date,
        start_date=start_date,
    )
    logger.debug("Processamento dos gribs finalizado")

    steps = list(nwp.keys())
    step_dfs = []

    for step in steps:

        if step not in states:
            logger.warning(f"Step {step} não encontrado nos estados salvos. Pulando.")
            continue

        logger.debug(f"Extraindo matrizes de features e de variável dependente para step = {step}")
        X, time_index = extract_grid_matrix(
            nwp_data=nwp[step],
            coordinates=[latitude, longitude + 360],
            n_points=CONFIG.data.n_points,
        )

        scaler = Scaler().from_state(states[step]["scaler"])
        
        X_scaled = scaler.forecast(X=X)

        df_step_preds = pd.DataFrame(index=time_index).reset_index()
        df_step_preds["horizon"] = step
        for name_model in models:
            
            if name_model not in states[step]:
                continue

            logger.debug(f"Carregando e executando predição para {name_model} no step = {step}")

            predict_module_path = f"downscaling_framework.core.{name_model}.predict"
            predict_module = importlib.import_module(predict_module_path)

            preds_scaled = predict_module.execute(X, states[step][name_model])

            if preds_scaled.ndim == 1:
                preds_scaled = preds_scaled.reshape(-1, 1)
            
            preds = scaler.inverse_transform(preds_scaled).flatten()

            df_step_preds[name_model] = preds

        step_dfs.append(df_step_preds)

    final_preds = pd.concat(step_dfs, ignore_index=True)


    return final_preds
    