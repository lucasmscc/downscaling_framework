import json
import pickle
from pathlib import Path

import pandas as pd
from utils import inmet_format

from downscaling_framework.train import execute as train_execute


def test_train_execution():
    obs_path = "./tests/data/input/obs"
    nwp_path = Path("./tests/data/input/gfs")
    config_path = "./tests/data/params/config"
    model_trained_path = "./tests/data/params/model_trained_state"

    id = "A319"
    target_variable = "wind_speed"
    release_range=(
        pd.Timestamp("2025-10-01 00:00:00"),
    )
    end_date = release_range[-1].strftime("%Y%m%d")

    with open(f"{config_path}/station_locations.json", encoding="utf-8") as file:
        station_locations = json.load(file)

    with open(f"{config_path}/gfs_config.json", encoding="utf-8") as file:
        gfs_config = json.load(file)

    latitude = station_locations[id]["lat"]
    longitude = station_locations[id]["lon"]

    df = pd.read_csv(
        f"{obs_path}/{id}.csv",
        sep=";",
        decimal=",",
    )
    df = inmet_format(df, "Vel. Vento (m/s)")
    df.rename(columns={"Vel": "wind_speed"}, inplace=True)

    states = train_execute(
        df=df,
        target_variable=target_variable,
        release_range=release_range,
        latitude=latitude,
        longitude=longitude,
        config=gfs_config,
        nwp_path=nwp_path,
    )

    with open(f"{model_trained_path}/params_{id}_{target_variable}_{end_date}.pkl", "wb") as f:
        pickle.dump(states, f)

    # Add any additional assertions or checks specific to train execution here
    assert True  # Placeholder assertion, replace with actual checks if needed
