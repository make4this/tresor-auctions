"""Page « History » : historique par maturité et par cible, avec les prévisions du backtest."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from tresor import app_data as ad
from tresor import app_ui as ui
from tresor.formats import en

st.title("History")
st.caption(
    "Every completed auction since 2009, the average of the last 3 auctions of the same"
    " maturity, and the out-of-sample forecast of the 5-model average (since 2014)."
)

features = ui.parquet("features.parquet", folder="clean")
backtest = ui.parquet("backtest_predictions.parquet")
left, middle, right = st.columns(3)
family = left.selectbox("Securities", list(ad.FAMILY_LABELS), format_func=ad.FAMILY_LABELS.get)
available = set(features.loc[features["family"] == family, "maturity_bucket"].dropna())
buckets = [bucket for bucket in ad.MATURITY_ORDER if bucket in available]
bucket = middle.selectbox("Maturity", buckets, format_func=ad.MATURITY_LABELS.get,
                          index=buckets.index("10Y") if "10Y" in buckets else 0)  # fmt: skip
target = right.selectbox("Measure", list(ad.TARGET_LABELS), format_func=ad.TARGET_LABELS.get)

series = ad.explorer_series(features, backtest, family, bucket, target)
share = target in ("dealer_share", "indirect_share")
scale, unit = (100, "%") if share else (1, " bp" if target == "tail_bp" else "")
digits = ".1f" if share or target == "tail_bp" else ".2f"
if series.empty:
    st.info("No completed auction with this measure for this maturity.")
else:
    dates = [ui.date_en(day) for day in series["auction_date"]]
    figure = go.Figure()
    figure.add_scatter(x=series["auction_date"], y=series["actual"] * scale,
                       mode="lines+markers", name="Actual", text=dates,
                       line={"color": ui.SERIES[0], "width": 1.5}, marker={"size": 5},
                       hovertemplate=f"%{{text}}<br>actual %{{y:{digits}}}{unit}"
                       "<extra></extra>")  # fmt: skip
    figure.add_scatter(x=series["auction_date"], y=series["mean3"] * scale, mode="lines",
                       name="Last-3 average", line={"color": ui.MUTED, "dash": "dot"},
                       hovertemplate=f"last-3 average %{{y:{digits}}}{unit}"
                       "<extra></extra>")  # fmt: skip
    predicted = series.dropna(subset=["prediction"])
    figure.add_scatter(x=predicted["auction_date"], y=predicted["prediction"] * scale,
                       mode="markers", name="Out-of-sample forecast (5-model average)",
                       marker={"color": ui.SERIES[1], "size": 6, "symbol": "diamond"},
                       hovertemplate=f"forecast %{{y:{digits}}}{unit}<extra></extra>")  # fmt: skip
    figure.update_layout(title=f"{ad.MATURITY_LABELS[bucket]}: {ad.TARGET_LABELS[target]}")
    figure.update_yaxes(ticksuffix=unit)
    figure.update_xaxes(rangeslider={"visible": True})
    ui.chart(figure, height=540)
    st.caption(
        f"{en(len(series), 0)} completed auctions, {en(len(predicted), 0)} of them forecast out"
        " of sample. Drag the bar below the chart to zoom on a period."
    )
