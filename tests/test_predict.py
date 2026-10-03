import json
import pickle
from pathlib import Path

import pandas as pd
from utils import get_params

from downscaling_framework.predict import execute as predict_execute


def test_predict_execution():

    nwp_path = Path("./tests/data/input/gfs")
    config_path = "./tests/data/params/config"
    model_trained_path = "./tests/data/params/model_trained_state"
    output_path = "./tests/data/output/predict"

    id = "A319"
    target_variable = "wind_speed"
    release_range=(
        pd.Timestamp("2025-10-01 00:00:00"),
        pd.Timestamp("2025-10-30 18:00:00")
    )
    end_date = release_range[1].strftime("%Y%m%d")

    with open(f"{config_path}/station_locations.json", encoding="utf-8") as file:
        station_locations = json.load(file)

    with open(f"{config_path}/gfs_config.json", encoding="utf-8") as file:
        gfs_config = json.load(file)

    states = get_params(model_trained_path, id, target_variable)

    latitude = station_locations[id]["lat"]
    longitude = station_locations[id]["lon"]

    preds = predict_execute(
        states=states,
        target_variable=target_variable,
        release_range=release_range,
        latitude=latitude,
        longitude=longitude,
        config=gfs_config,
        nwp_path=nwp_path,
    )

    preds.to_parquet(f"{output_path}/ds_gfs_{id}_{target_variable}_{end_date}.parquet")

    # Add any additional assertions or checks specific to predict execution here
    assert True  # Placeholder assertion, replace with actual checks if needed
