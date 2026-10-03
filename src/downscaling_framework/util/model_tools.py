import math
from typing import Optional, Tuple
import numpy as np
import pandas as pd

def convert_cartesian_to_polar(
    u: np.ndarray, 
    v: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Converte vetores cartesianos de vento (u, v) para velocidade e direção polar.

    Args:
        u (np.ndarray): Vetor da componente u do vento.
        v (np.ndarray): Vetor da componente v do vento.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Vetores de velocidade (speed) e direção (direction).
    """
    u_vals = np.asarray(u, dtype=float)
    v_vals = np.asarray(v, dtype=float)

    direction = (270 - np.degrees(np.arctan2(v_vals, u_vals))) % 360
    speed = np.sqrt(u_vals**2 + v_vals**2)

    return speed, direction


def extract_grid_matrix(
    nwp_data: pd.DataFrame,
    coordinates: list[float],
    n_points: int,
    observations: pd.DataFrame = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], pd.MultiIndex]:
    """Extrai a grade do NWP mantendo DateTime(release) e DateTime(forecast) como MultiIndex.

    Args:
        nwp_data (pd.DataFrame): DataFrame contendo os dados do NWP.
        coordinates (list[float]): Lista contendo [latitude, longitude] alvo.
        n_points (int): Número total de pontos da grade NWP.
        observations (pd.Series, optional): Série temporal de observações (alvo).

    Returns:
        Tuple[np.ndarray, Optional[np.ndarray], pd.MultiIndex]: 
            Matriz X, vetor y e o MultiIndex contendo (release, forecast).
    """

    if {"u", "v"}.issubset(nwp_data.columns):
        speed, _ = convert_cartesian_to_polar(
            nwp_data["u"].to_numpy(), 
            nwp_data["v"].to_numpy()
        )
        
        nwp_data["wind_speed"] = speed

    latitude, longitude = coordinates
    grid_size = int(math.sqrt(n_points) / 2)

    latitudes = sorted(list(set(nwp_data["lat"])) + [latitude])
    longitudes = sorted(list(set(nwp_data["lon"])) + [longitude])

    lat_idx = latitudes.index(latitude)
    lon_idx = longitudes.index(longitude)

    grid_latitudes = (
        latitudes[lat_idx - grid_size : lat_idx]
        + latitudes[lat_idx + 1 : lat_idx + grid_size + 1]
    )
    grid_longitudes = (
        longitudes[lon_idx - grid_size : lon_idx]
        + longitudes[lon_idx + 1 : lon_idx + grid_size + 1]
    )

    mesh = np.array(np.meshgrid(grid_latitudes, grid_longitudes)).T.reshape(-1, 2)
    
    data_frames = []

    for i, (lat, lon) in enumerate(mesh):
        subset = nwp_data.loc[(nwp_data["lat"] == lat) & (nwp_data["lon"] == lon)]
        if i == 0:
            columns = [
                "DateTime(release)",
                "DateTime(forecast)",
                "wind_speed",
            ]
            renamed_columns = [
                "DateTime(release)",
                "DateTime(forecast)",
                f"wind_speed_{i}",
            ]
        else:
            columns = ["wind_speed"]
            renamed_columns = [f"wind_speed_{i}"]
        subset = subset[columns].reset_index(drop=True)
        subset.columns = renamed_columns
        data_frames.append(subset)

    merged_grid = pd.concat(data_frames, axis=1)
    merged_grid["DateTime(release)"] = pd.to_datetime(
        merged_grid["DateTime(release)"],
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce",
    )
    merged_grid["DateTime(forecast)"] = pd.to_datetime(
        merged_grid["DateTime(forecast)"],
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce",
    )

    if observations is None:
        aligned_data = merged_grid.set_index(["DateTime(release)", "DateTime(forecast)"])
        time_index = aligned_data.index
        
        X = aligned_data.to_numpy(dtype=float)

        return X, time_index
    else:
        merged_data = (
            pd.merge(
                merged_grid,
                observations,
                how="inner",
                left_on="DateTime(forecast)",
                right_on="datetime",
            )
            .dropna()
            .reset_index(drop=True)
        )

        aligned_data = merged_data.set_index(["DateTime(release)", "DateTime(forecast)"])
        aligned_data = aligned_data.drop(columns=["datetime"])

        time_index = aligned_data.index
        X = aligned_data.iloc[:, :-1].to_numpy(dtype=float)
        y = aligned_data.iloc[:, -1].to_numpy(dtype=float)

        return X, y, time_index