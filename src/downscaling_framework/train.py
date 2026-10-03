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
    df: pd.DataFrame,
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
        df (DataFrame):
            DataFrame com os dados observados. 
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

    states = {}
    for step in steps:
        states[step] = {}
        # Extrai a matriz X e o vetor y unidimensional
        logger.debug(f"Extraindo matrizes de features e de variável dependente para step = {step}")
        X, y, _ = extract_grid_matrix(
            nwp_data=nwp[step],
            coordinates=[latitude, longitude + 360],
            n_points=CONFIG.data.n_points,
            observations=df,
        )

        logger.debug("Split dos dados")
        # Realiza o split dos dados (ex: 80% treino, 20% validação)
        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=CONFIG.data.test_size, random_state=42, shuffle=False
        )

        logger.debug("Reshape dos vetores unidimensionais")
        y_train_arr = (
            y_train.to_numpy() if hasattr(y_train, "to_numpy") else np.asarray(y_train)
        ).reshape(-1, 1)
        y_valid_arr = (
            y_valid.to_numpy() if hasattr(y_valid, "to_numpy") else np.asarray(y_valid)
        ).reshape(-1, 1)

        logger.debug(f"Normalizando dados do step {step}")
        scaler = Scaler()
        X_train_scaled, y_train_scaled = scaler.train(X_train, y_train_arr)
        X_valid_scaled, y_valid_scaled = scaler.forecast(X_valid, y_valid_arr)

        states[step]["scaler"] = scaler.get_state()
        
        train_params = asdict(CONFIG.training)

        for name_model in models:

            logger.debug(f"Iniciando treinamento para {name_model} em step = {step}")

            model_params = asdict(CONFIG.models[name_model])
            model_configs = {**train_params, **model_params}
            
            # Importa dinamicamente o módulo de treino do modelo (ex: cnn, mlp)
            train_module_path = f"downscaling_framework.core.{name_model}.train"
            train_module = importlib.import_module(train_module_path)
            
            model_params = asdict(CONFIG.models[name_model])
            model_configs = {**train_params, **model_params}

            logger.debug(f"Parâmetros de configuração carregados para {name_model} em step = {step}")
            
            # Executa o treinamento passando os arrays e os scalers adaptados
            model = train_module.execute(
                X_train=X_train_scaled,
                y_train=y_train_scaled,
                X_valid=X_valid_scaled,
                y_valid=y_valid_scaled,
                configs=model_configs,
            )

            logger.debug(f"Treinamento concluído para {name_model} em step = {step}")
            # Salva o estado do modelo treinado
            states[step][name_model] = model.get_state()

    return states
