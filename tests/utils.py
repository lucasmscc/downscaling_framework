import glob
import os
import pickle
import re

import pandas as pd


def inmet_format(df: pd.DataFrame, target_column: str):
    """
    Padroniza os dados de observação.

    Args:
        df (DataFrame):
            DataFrame com os dados observados. 
        target_column (str):
            Nome da coluna alvo para previsão.

    """

    df["datetime"] = pd.to_datetime(
        df["Data"].astype(str).str.strip()
        + " "
        + df["Hora (UTC)"].astype(str).str.zfill(4),
        format="%d/%m/%Y %H%M",
    )

    # df[target_column] = df[target_column].fillna(df[target_column].mean())
    # df[target_column] = df[target_column].interpolate()
    df = df.dropna(subset=[target_column])

    return df[["datetime", target_column]]


def get_params(model_trained_path, id, target_variable):
    pattern = os.path.join(
        model_trained_path,
        f"params_{id}_{target_variable}_*.pkl"
    )

    files = glob.glob(pattern)

    if not files:
        raise FileNotFoundError(
            f"Nenhum arquivo de parâmetros encontrado para "
            f"id={id} e target_variable={target_variable}"
        )

    # Extrai a data YYYYMMDD do nome do arquivo
    def extract_date(filepath):
        filename = os.path.basename(filepath)
        match = re.search(r"_(\d{8})\.pkl$", filename)

        if not match:
            raise ValueError(
                f"Data não encontrada no nome do arquivo: {filename}"
            )

        return match.group(1)

    # Seleciona o arquivo com a data mais recente
    latest_file = max(files, key=extract_date)

    with open(latest_file, "rb") as file:
        states = pickle.load(file)

    return states