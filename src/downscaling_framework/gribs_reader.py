import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import PosixPath
from typing import Literal

import numpy as np
import pandas as pd
import pygrib
import xarray as xr

def get_grid(
    latitude: float,
    longitude: float,
    latitudes_grid: np.ndarray,
    longitudes_grid: np.ndarray,
    N_POINTS: int = 4,
) -> tuple[slice]:
    """
    Extrai os índices dos quatro pontos mais próximos do NWP.

    Args:
        latitude (float):
            Latitude de interesse.
        longitude (float):
            Longitude de interesse.
        latitudes_grid (float):
            Matriz com as latitudes do NWP.
        longitudes_grid (float):
            Matriz com as longitudes do NWP.
        N_POINTS (int, optional):
            Número de pontos acima e abaixo das coordenadas de interesse.
            Defaults to 4 (grid 8x8).

    Returns:
        tuple[slice]: índices da malha de pontos do NWP.

    """
    grib_max_lon = longitudes_grid.max()
    lon_search = (
        longitude + 360.0
        if (longitude < 0 and grib_max_lon > 180)
        else longitude
    )

    lats_1d = latitudes_grid[:, 0]
    lons_1d = longitudes_grid[0, :]

    row_idx = np.searchsorted(
        lats_1d if lats_1d[0] < lats_1d[-1] else lats_1d[::-1],
        latitude,
    )
    col_idx = np.searchsorted(lons_1d, lon_search)

    if lats_1d[0] > lats_1d[-1]:
        row_idx = len(lats_1d) - row_idx

    lat_start = max(
        0,
        min(row_idx - N_POINTS, latitudes_grid.shape[0] - N_POINTS * 2),
    )
    lon_start = max(0, min(col_idx - N_POINTS, longitudes_grid.shape[1] - 8))

    return (
        slice(lat_start, lat_start + (N_POINTS * 2), 1),
        slice(lon_start, lon_start + (N_POINTS * 2), 1),
    )


def filter_files_by_period(
    base_dir: PosixPath,
    reference_date: datetime,
    start_date: datetime | None = None,
) -> list[str]:
    """
    Filtro dos arquivos para o período desejado.

    Args:
        base_dir (PosixPath):
            Diretório dos arquivos NWP
        reference_date (str):
            Timestamp de referência para execução.
        start_date (str | None, optional):
            Timestamp inicial para o modo de treinamento. Defaults to None.

    Returns:
        list[str]:
            Lista contendo todos os arquivos desejado.

    """
    items = base_dir.rglob("*.grib2")
    re_file = r"gfs\.t(?P<rel>\d{2})z_(?P<date>\d{8})_H\d+\.grib2"

    pattern = re.compile(re_file)

    parsed_data = []
    for item in items:
        match = pattern.search(item.name)

        if not match:
            continue

        try:
            date_str = match.group("date")

            dt = datetime(
                int(date_str[0:4]),
                int(date_str[4:6]),
                int(date_str[6:8]),
            )

            release = int(match.group("rel"))
        except ValueError:
            continue

        # Filtragem antecipada
        if start_date is not None:
            if not (start_date <= dt <= reference_date):
                continue
        elif dt != reference_date or release != reference_date.hour:
            continue

        parsed_data.append((dt, release, str(item)))

    parsed_data.sort()

    return [path for _, _, path in parsed_data]


def extract_gribs(  # noqa: PLR0913
    file: PosixPath,
    configs: dict,
    level: int,
    idx_lat: int,
    idx_lon: int,
    lats_sub: np.ndarray,
    lons_sub: np.ndarray,
) -> dict:
    """Worker paralelo para leitura de um arquivo GRIB2 (nível único)."""
    needed_keys = {
        (msg_name, lvl_type, int(level))
        for msg_name, lvl_type in configs["msgs"]
    }
    storage = dict()

    with pygrib.open(file) as grb:
        sample = grb.message(1)
        ref_time = sample.analDate
        step = float(sample.endStep)
        f_time = ref_time + pd.Timedelta(hours=step)

        grb.seek(0)
        all_messages = dict()
        for msg in grb:
            key = (msg.name, msg.typeOfLevel, msg.level)

            if key not in needed_keys:
                continue

            all_messages[key] = msg.values[idx_lat, idx_lon].flatten()

        row = {
            "DateTime(release)": ref_time,
            "DateTime(forecast)": f_time,
            "lat": lats_sub,
            "lon": lons_sub,
        }

        for (msg_name, lvl_type), col_name in zip(
            configs["msgs"],
            configs["cols"],
            strict=False,
        ):
            key = (msg_name, lvl_type, int(level))
            row[col_name] = all_messages[key]

        if step not in storage:
            storage[step] = list()
        storage[step].append(row)

    return storage

def merge_grib_storage(
    global_storage: dict,
    partial_storage: dict,
) -> None:
    """Junta resultados dos workers para estrutura de nível único."""
    for step, rows in partial_storage.items():
        if step not in global_storage:
            global_storage[step] = []

        global_storage[step].extend(rows)


def process_gribs(
    configs: dict,
    root_directory: PosixPath,
    coordinates: tuple[float],
    end_date: str,
    start_date: str | None = None,
) -> dict:
    """Processamento dos arquivos '*.grib' para um único nível."""
    
    # Considera apenas o primeiro nível configurado
    level = configs["preassure_level"]
    
    latitude, longitude = coordinates

    files = filter_files_by_period(
        root_directory,
        end_date,
        start_date,
    )

    if not files:
        return None

    # Calcula grid apenas uma vez
    with pygrib.open(files[0]) as grb:
        sample = grb.message(1)

        lats, lons = sample.latlons()

        idx_lat, idx_lon = get_grid(
            latitude,
            longitude,
            lats,
            lons,
        )

        lats_sub = lats[idx_lat, idx_lon].flatten()
        lons_sub = lons[idx_lat, idx_lon].flatten()

    # Processamento paralelo
    storage = dict()
    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                extract_gribs,
                file,
                configs,
                level,
                idx_lat,
                idx_lon,
                lats_sub,
                lons_sub,
            )
            for file in files
        ]

        for future in as_completed(futures):
            try:
                partial_storage = future.result()
                merge_grib_storage(
                    storage,
                    partial_storage,
                )
            except Exception as e:
                print(
                    f"Falha ao processar arquivo, ignorando: {e}"
                )
                continue

    # Conversão final (removendo o nível do loop)
    data_store = {}
    for step, rows in storage.items():
        df = pd.DataFrame(rows)
        df = df.explode(
            column=[
                "lat",
                "lon",
                *configs["cols"],
            ],
            ignore_index=True,
        )
        data_store[step] = df

    print("Extração de dados do GFS concluída!")
    return data_store