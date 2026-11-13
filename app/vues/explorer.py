"""Page 4 : historique par maturité et par cible, avec les prévisions du backtest."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from tresor import app_data as ad
from tresor import app_ui as ui

st.title("Explorer")
st.caption(
    "Historique des enchères réalisées depuis 2009 (l'échantillon des modèles), la moyenne des 3"
    " dernières de même maturité, et la prévision hors échantillon de la moyenne des 5 modèles"
    " (depuis 2014)."
)

features = ui.parquet("features.parquet", folder="clean")
backtest = ui.parquet("backtest_predictions.parquet")
left, middle, right = st.columns(3)
family = left.selectbox("Famille", list(ad.FAMILY_LABELS), format_func=ad.FAMILY_LABELS.get)
available = set(features.loc[features["family"] == family, "maturity_bucket"].dropna())
buckets = [bucket for bucket in ad.MATURITY_ORDER if bucket in available]
bucket = middle.selectbox("Maturité", buckets, format_func=ad.MATURITY_LABELS.get,
                          index=buckets.index("10Y") if "10Y" in buckets else 0)  # fmt: skip
target = right.selectbox("Cible", list(ad.TARGET_LABELS), format_func=ad.TARGET_LABELS.get)

series = ad.explorer_series(features, backtest, family, bucket, target)
share = target in ("dealer_share", "indirect_share")
scale, unit = (100, " %") if share else (1, " pb" if target == "tail_bp" else "")
digits = ".1f" if share or target == "tail_bp" else ".2f"
if series.empty:
    st.info("Pas d'enchère réalisée avec cette donnée pour cette maturité.")
else:
    dates = [ui.date_fr(day) for day in series["auction_date"]]
    figure = go.Figure()
    figure.add_scatter(x=series["auction_date"], y=series["actual"] * scale,
                       mode="lines+markers", name="Réalisé", text=dates,
                       line={"color": ui.SERIES[0], "width": 1.5}, marker={"size": 5},
                       hovertemplate=f"%{{text}}<br>réalisé %{{y:{digits}}}{unit}"
                       "<extra></extra>")  # fmt: skip
    figure.add_scatter(x=series["auction_date"], y=series["mean3"] * scale, mode="lines",
                       name="Moyenne des 3 dernières", line={"color": ui.MUTED, "dash": "dot"},
                       hovertemplate=f"moyenne des 3 %{{y:{digits}}}{unit}"
                       "<extra></extra>")  # fmt: skip
    predicted = series.dropna(subset=["prediction"])
    figure.add_scatter(x=predicted["auction_date"], y=predicted["prediction"] * scale,
                       mode="markers", name="Prévision du backtest (moyenne des 5)",
                       marker={"color": ui.SERIES[1], "size": 6, "symbol": "diamond"},
                       hovertemplate=f"prévu %{{y:{digits}}}{unit}<extra></extra>")  # fmt: skip
    figure.update_layout(title=f"{ad.MATURITY_LABELS[bucket]} : {ad.TARGET_LABELS[target]}")
    figure.update_yaxes(ticksuffix=unit)
    figure.update_xaxes(rangeslider={"visible": True})
    ui.chart(figure, height=540)
    st.caption(
        f"{ad.fr(len(series), 0)} enchères réalisées, dont {ad.fr(len(predicted), 0)} prévues"
        " hors échantillon. Faire glisser la barre du bas pour zoomer sur une période."
    )
