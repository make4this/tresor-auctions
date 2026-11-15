"""Page « Live track record » : chaque prévision figée face au résultat publié."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from tresor import app_data as ad
from tresor import app_ui as ui

st.title("Live track record")
st.caption(
    "Each frozen forecast is compared with the result published by the Treasury, and with the"
    " average of the last 3 auctions of the same maturity, the benchmark to beat."
)

tracking = ui.parquet("live_tracking.parquet")
counts = ad.live_summary(tracking)
columns = st.columns(4)
columns[0].metric("Auctions tracked", counts["suivies"])
columns[1].metric("Results in", counts["realisees"])
columns[2].metric("Waiting", counts["en_attente"])
columns[3].metric("No result", counts["sans_resultat"], help="Date passed without a result:"
                  " auction moved or cancelled, or result not collected yet.")  # fmt: skip

target = st.segmented_control(
    "Measure", list(ad.TARGET_LABELS), default="btc", format_func=ad.TARGET_LABELS.get,
    required=True,
) or "btc"  # fmt: skip
scale = 100 if target in ("dealer_share", "indirect_share") else 1
done = ad.live_results(tracking, target)


def shown(values: pd.Series, sign: bool = False) -> pd.Series:
    return values.map(lambda v: ad.format_target(v, target, sign=sign))


if done.empty:
    pending = tracking[(tracking["model"] == "moyenne_5") & (tracking["status"] == "en_attente")]
    if counts["realisees"]:
        st.info("Some auctions have taken place, but no result is published for this measure.")
    else:
        first = pending["auction_date"].min()
        when = f" ({ui.date_en(first)})" if pd.notna(first) else ""
        st.info(
            f"No tracked auction has taken place yet. Results will appear here after the first"
            f" one{when}, with the evening update. *Model performance* shows the same comparison"
            " over 2014-2026."
        )
else:
    left, right = st.columns(2)
    figure = go.Figure()
    low = min(done["actual"].min(), done["pred"].min()) * scale
    high = max(done["actual"].max(), done["pred"].max()) * scale
    figure.add_scatter(x=[low, high], y=[low, high], mode="lines", name="Perfect forecast",
                       line={"color": ui.MUTED, "dash": "dot", "width": 1})  # fmt: skip
    figure.add_scatter(
        x=done["pred"] * scale,
        y=done["actual"] * scale,
        mode="markers",
        name="5-model average",
        marker={"color": ui.SERIES[0], "size": 9},
        text=[ui.date_en(day) for day in done["auction_date"]],
        hovertemplate="%{text}<br>forecast %{x:.2f}<br>actual %{y:.2f}<extra></extra>",
    )
    figure.add_scatter(
        x=done["bench_mean3"] * scale,
        y=done["actual"] * scale,
        mode="markers",
        name="Last-3 average",
        marker={"color": ui.SERIES[1], "size": 8, "symbol": "diamond"},
        hovertemplate="last-3 average %{x:.2f}<br>actual %{y:.2f}<extra></extra>",
    )
    unit = " (%)" if scale == 100 else (" (bp)" if target == "tail_bp" else "")
    figure.update_layout(title="Forecast versus actual")
    figure.update_xaxes(title=f"Forecast{unit}")
    figure.update_yaxes(title=f"Actual{unit}")
    ui.chart(figure, container=left)
    figure = go.Figure()
    line = {"color": ui.SERIES[0], "width": 2}
    figure.add_scatter(x=done["auction_date"], y=done["gain"] * 100, mode="lines+markers",
                       name="Cumulative gain", line=line)  # fmt: skip
    figure.update_layout(title="Cumulative gain versus the last-3 average", showlegend=False)
    figure.update_yaxes(ticksuffix="%")
    ui.chart(figure, container=right)
    st.caption(
        "Cumulative gain: how much smaller the forecast's squared errors are than those of the"
        " last-3 average, as a share of the latter. Above 0, the forecast does better."
    )
    st.dataframe(
        pd.DataFrame(
            {
                "Date": [ui.date_en(day) for day in done["auction_date"]],
                "Auction": [ad.MATURITY_LABELS.get(b, b) for b in done["maturity_bucket"]],
                "Forecast": shown(done["pred"]),
                "Last-3 average": shown(done["bench_mean3"]),
                "Actual": shown(done["actual"]),
                "Forecast error": shown(done["error"], sign=True),
                "Last-3 average error": shown(done["bench_error"], sign=True),
            }
        ),
        hide_index=True,
        width="stretch",
    )

waiting = tracking[(tracking["model"] == "moyenne_5") & (tracking["status"] != "realisee")]
waiting = waiting[waiting["target"] == target].sort_values("auction_date")
if not waiting.empty:
    st.subheader("Forecasts waiting for a result")
    st.dataframe(
        pd.DataFrame(
            {
                "Date": [ui.date_en(day) for day in waiting["auction_date"]],
                "Auction": [ad.MATURITY_LABELS.get(b, b) for b in waiting["maturity_bucket"]],
                "Forecast": shown(waiting["pred"]),
                "Last-3 average": shown(waiting["bench_mean3"]),
                "Status": waiting["status"].map(
                    {"en_attente": "waiting", "sans_resultat": "no result"}
                ),
                "Forecast made on": [
                    ui.date_en(t.tz_convert("America/New_York")) for t in waiting["predicted_at"]
                ],
            }
        ),
        hide_index=True,
        width="stretch",
    )
