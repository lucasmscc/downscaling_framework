import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

FONT = "Arial"
BG_PLOT = "#F8F9FA"
BG_PAPER = "#FFFFFF"

METRIC_LABELS = {
    "corr": "Correlação",
    "std_ratio": "Razão do Desvio Padrão",
    "bias": "Bias",
    "rmse": "RMSE",
}

def calculate_metrics(
    df: pd.DataFrame,
    preds: pd.DataFrame,
    forecast_range: tuple
):
    """
    Realiza o filtro por período, o merge entre observações e previsões,
    e calcula as 4 métricas por horizonte e modelo:
    Correlação, Std Ratio, Bias e RMSE.
    """

    start_date, end_date = forecast_range

    mask = (
        (df["datetime"] >= start_date)
        & (df["datetime"] <= end_date)
    )

    df_filtered = df.loc[mask].copy()

    models = list(preds.columns)[3:]

    target_column = [
        col for col in df.columns
        if col != "datetime"
    ][0]

    metrics = {
        "corr": {},
        "std_ratio": {},
        "bias": {},
        "rmse": {}
    }

    for horizon in sorted(set(preds["horizon"])):

        preds_h = preds.query(
            f"horizon == {horizon}"
        )

        merged_df = pd.merge(
            df_filtered,
            preds_h,
            left_on="datetime",
            right_on="DateTime(forecast)"
        ).dropna()

        if merged_df.empty:
            continue

        obs = merged_df[target_column]

        metrics["corr"][horizon] = {}
        metrics["std_ratio"][horizon] = {}
        metrics["bias"][horizon] = {}
        metrics["rmse"][horizon] = {}

        for model in models:

            pred = merged_df[model]

            metrics["corr"][horizon][model] = (
                obs.corr(pred)
            )

            obs_std = obs.std()
            pred_std = pred.std()

            metrics["std_ratio"][horizon][model] = (
                pred_std / obs_std
                if obs_std != 0
                else np.nan
            )

            metrics["bias"][horizon][model] = (
                pred - obs
            ).mean()

            metrics["rmse"][horizon][model] = np.sqrt(
                np.mean((pred - obs) ** 2)
            )

    return {
        key: pd.DataFrame(val).T
        for key, val in metrics.items()
    }

def plot_metrics(metrics_dict: dict):
    """
    Plota as 4 métricas em uma matriz 2x2 usando Plotly.

    Inclui:
    - linhas + marcadores;
    - cores fixas por modelo;
    - legenda compartilhada;
    - tabela com a média das métricas;
    - zoom e interação do Plotly.
    """

    metrics = [
        "corr",
        "std_ratio",
        "bias",
        "rmse"
    ]

    dfs = {
        metric: metrics_dict[metric]
        for metric in metrics
        if metric in metrics_dict
        and not metrics_dict[metric].empty
    }

    if not dfs:
        print("Nenhuma métrica disponível para plotagem.")
        return None

    models = set()

    for df_metric in dfs.values():
        models.update(df_metric.columns)

    models = sorted(models)

    palette = px.colors.qualitative.Plotly

    color_map = {
        model: palette[i % len(palette)]
        for i, model in enumerate(models)
    }

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "(a) Correlação",
            "(b) Razão do Desvio Padrão",
            "(c) Bias",
            "(d) RMSE",
        ],
        shared_xaxes=True,
        vertical_spacing=0.10,
        horizontal_spacing=0.08,
    )

    positions = {
        "corr": (1, 1),
        "std_ratio": (1, 2),
        "bias": (2, 1),
        "rmse": (2, 2),
    }

    for metric, df_metric in dfs.items():

        row, col = positions[metric]

        for model in models:

            if model not in df_metric.columns:
                continue

            fig.add_trace(
                go.Scatter(
                    x=df_metric.index,
                    y=df_metric[model],
                    name=model,
                    legendgroup=model,
                    showlegend=(metric == "corr"),
                    line=dict(
                        color=color_map[model],
                        width=2.5,
                    ),
                    marker=dict(
                        size=6
                    ),
                    hovertemplate=(
                        f"<b>{model}</b><br>"
                        "Horizonte: %{x}<br>"
                        "Valor: %{y:.4f}"
                        "<extra></extra>"
                    ),
                ),
                row=row,
                col=col,
            )

    fig.update_xaxes(
        title_text="Horizonte",
        showgrid=True,
        gridcolor="rgba(0,0,0,0.10)",
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(0,0,0,0.10)",
    )

    # Correlação entre -1 e 1
    fig.update_yaxes(
        range=[-1, 1],
        row=1,
        col=1,
    )

    fig.update_yaxes(
        title_text="Correlação de Pearson",
        row=1,
        col=1,
    )

    fig.update_yaxes(
        title_text="Std Ratio",
        row=1,
        col=2,
    )

    fig.update_yaxes(
        title_text="Bias",
        row=2,
        col=1,
    )

    fig.update_yaxes(
        title_text="RMSE",
        row=2,
        col=2,
    )

    fig.update_layout(
        title=dict(
            text="Métricas por Horizonte",
            font=dict(
                size=30,
                family=FONT,
            ),
            x=0.5,
            xanchor="center",
        ),

        font=dict(
            family=FONT,
            size=18,
        ),

        plot_bgcolor=BG_PLOT,
        paper_bgcolor=BG_PAPER,

        legend=dict(
            orientation="v",
            x=1.02,
            xanchor="left",
            y=1,
            yanchor="top",
            font=dict(
                size=18,
                family=FONT,
            ),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.1)",
            borderwidth=1,
        ),

        width=1500,
        height=950,

        margin=dict(
            l=80,
            r=180,
            t=120,
            b=80,
        ),

        hovermode="x unified",

        template="simple_white",
    )

    for annotation in fig.layout.annotations:
        annotation.font.size = 20
        annotation.font.color = "#555555"


    fig.show()

    return fig


def main(
    df: pd.DataFrame,
    preds: pd.DataFrame,
    forecast_range: tuple
):
    """
    Função principal.

    Os argumentos permanecem exatamente os mesmos:

        main(df, preds, forecast_range)
    """

    print(
        "Calculando métricas para o período selecionado..."
    )

    metrics_dict = calculate_metrics(
        df,
        preds,
        forecast_range
    )

    print(
        "Gerando visualizações com Plotly..."
    )

    fig = plot_metrics(metrics_dict)

    return metrics_dict
