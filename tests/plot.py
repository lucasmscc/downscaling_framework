import pandas as pd
from evaluation.main import main as main_execute
from utils import inmet_format

id = "A319"
target_variable = "wind_speed"
forecast_range = (
    pd.Timestamp("2025-10-01 00:00:00"),
    pd.Timestamp("2025-10-30 23:00:00")
)

df = pd.read_csv(
    f"./tests/data/input/obs/{id}.csv",
    sep=";",
    decimal=",",
)
df = inmet_format(df, "Vel. Vento (m/s)")
df.rename(columns={"Vel": "wind_speed"}, inplace=True)

end_date = forecast_range[-1].strftime("%Y%m%d")
preds = pd.read_parquet(f"./tests/data/output/predict/ds_gfs_{id}_{target_variable}_{end_date}.parquet")


main_execute(
    df=df,
    preds=preds,
    forecast_range=forecast_range
) 