"""Page « Live track record » : chaque prévision figée face au résultat publié."""

from __future__ import annotations

from html import escape

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from tresor import app_data as ad
from tresor import app_ui as ui
from tresor.formats import en

ui.page_header(
    "Live track record",
    "Each forecast, frozen on auction day, is compared with the result published by the"
    " Treasury and with the average of the last 3 auctions of the same maturity, the benchmark"
    " to beat.",
)

tracking = ui.parquet("live_tracking.parquet")
counts = ad.live_summary(tracking)
st.markdown(
    ui.panel(
        "Tracking so far",
        ui.stats(
            [
                (en(counts["suivies"], 0), "auctions tracked"),
                (en(counts["realisees"], 0), "results in"),
                (en(counts["en_attente"], 0), "waiting"),
                (en(counts["sans_resultat"], 0), "no result"),
            ]
        ),
        guide=[
            "<b>Tracked</b>: auctions with a forecast frozen on auction day.",
            "<b>Results in</b>: the Treasury has published the result.",
            "<b>Waiting</b>: the auction has not taken place yet, or its result is not"
            " collected yet (evening update).",
            "<b>No result</b>: date passed without a result: auction moved or cancelled, or"
            " result not collected yet.",
        ],
    ),
    unsafe_allow_html=True,
)

target = st.segmented_control(
    "Measure", list(ad.TARGET_LABELS), default="btc", format_func=ad.TARGET_LABELS.get,
    required=True,
) or "btc"  # fmt: skip
scale = 100 if target in ("dealer_share", "indirect_share") else 1
unit = " (%)" if scale == 100 else (" (bp)" if target == "tail_bp" else "")
done = ad.live_results(tracking, target)


def shown(value: float, sign: bool = False) -> str:
    return ad.format_target(value, target, sign=sign)


CLOSER_TAGS = {
    ad.FORECAST: '<span class="tr-tag tr-strong">Forecast</span>',
    ad.LAST3: '<span class="tr-tag tr-normal">Last 3</span>',
    ad.TIE: '<span class="tr-tag tr-normal">Tie</span>',
}

if done.empty:
    if counts["realisees"]:
        message = "Some auctions have taken place, but no result is published for this measure."
    else:
        pending = tracking[
            (tracking["model"] == "moyenne_5") & (tracking["status"] == "en_attente")
        ]
        first = pending["auction_date"].min()
        when = f" ({ui.date_en(first)})" if pd.notna(first) else ""
        message = (
            f"No tracked auction has taken place yet. Results will appear here after the first"
            f" one{when}, with the evening update. The Model performance page shows the same"
            " comparison over 2014-2026."
        )
    st.markdown(
        ui.panel("Results", f'<div class="tr-panel-note">{escape(message)}</div>'),
        unsafe_allow_html=True,
    )
else:
    left, right = st.columns(2)
    with (
        left,
        ui.box(
            "live_scatter",
            "Forecast versus actual",
            [
                "Each point is one auction: forecast across, actual result up.",
                "On the dotted diagonal the forecast was exact; the closer, the better.",
                "<b>Blue dots</b>: 5-model average. <b>Orange diamonds</b>: last-3 average.",
            ],
        ),
    ):
        figure = go.Figure()
        low = min(done["actual"].min(), done["pred"].min(), done["bench_mean3"].min()) * scale
        high = max(done["actual"].max(), done["pred"].max(), done["bench_mean3"].max()) * scale
        figure.add_scatter(x=[low, high], y=[low, high], mode="lines", name="Perfect forecast",
                           line={"color": ui.MUTED, "dash": "dot", "width": 1})  # fmt: skip
        figure.add_scatter(
            x=done["pred"] * scale, y=done["actual"] * scale, mode="markers",
            name="5-model average", marker={"color": ui.SERIES[0], "size": 9},
            text=[ui.date_en(day) for day in done["auction_date"]],
            hovertemplate="%{text}<br>forecast %{x:.2f}<br>actual %{y:.2f}<extra></extra>",
        )  # fmt: skip
        figure.add_scatter(
            x=done["bench_mean3"] * scale, y=done["actual"] * scale, mode="markers",
            name="Last-3 average",
            marker={"color": ui.SERIES[1], "size": 8, "symbol": "diamond"},
            hovertemplate="last-3 average %{x:.2f}<br>actual %{y:.2f}<extra></extra>",
        )  # fmt: skip
        figure.update_xaxes(title=f"Forecast{unit}")
        figure.update_yaxes(title=f"Actual{unit}")
        ui.chart(figure, height=360)
    with (
        right,
        ui.box(
            "live_gain",
            "Cumulative gain versus the last-3 average",
            [
                "Running total of how much smaller the forecast's squared errors are than those of"
                " the last-3 average, as a share of the latter.",
                "Above 0, the forecast does better so far; the last point is the live R².",
            ],
        ),
    ):
        figure = go.Figure()
        figure.add_scatter(x=done["auction_date"], y=done["gain"] * 100, mode="lines+markers",
                           name="Cumulative gain",
                           line={"color": ui.SERIES[0], "width": 2})  # fmt: skip
        figure.update_layout(showlegend=False)
        figure.update_yaxes(ticksuffix="%")
        ui.chart(figure, height=360)

    newest = done.iloc[::-1]  # les plus récentes d'abord
    decimals = 2 if target == "btc" else 1  # précision affichée (parts en %, tail en bp)
    winners = [
        ad.closer(e * scale, b * scale, decimals)
        for e, b in zip(newest["error"], newest["bench_error"], strict=True)
    ]
    rows = [
        [
            escape(ui.date_en(row.auction_date)),
            f"<b>{escape(ad.MATURITY_LABELS.get(row.maturity_bucket, row.maturity_bucket))}</b>",
            shown(row.pred),
            shown(row.bench_mean3),
            shown(row.actual),
            shown(row.error, sign=True),
            shown(row.bench_error, sign=True),
            CLOSER_TAGS[winner],
        ]
        for row, winner in zip(newest.itertuples(), winners, strict=True)
    ]
    wins = winners.count(ad.FORECAST)
    st.markdown(
        ui.panel(
            "Results",
            ui.table(
                [
                    "Date",
                    "Auction",
                    "Forecast",
                    "Last 3",
                    "Actual",
                    "Forecast error",
                    "Last-3 error",
                    "Closer",
                ],
                rows,
                right={2, 3, 4, 5, 6},
                scroll=True,
            ),  # fmt: skip
            f"The forecast was closer to the result than the last-3 average for {en(wins, 0)} of"
            f" {en(len(rows), 0)} auctions.",
            [
                "<b>Error</b> = actual − forecast (or actual − last-3 average).",
                "<b>Closer</b>: which of the two landed nearer the published result.",
            ],
        ),
        unsafe_allow_html=True,
    )

waiting = tracking[(tracking["model"] == "moyenne_5") & (tracking["status"] != "realisee")]
waiting = waiting[waiting["target"] == target].sort_values("auction_date")
if not waiting.empty:
    status = {"en_attente": "waiting", "sans_resultat": "no result"}
    rows = [
        [
            escape(ui.date_en(row.auction_date)),
            f"<b>{escape(ad.MATURITY_LABELS.get(row.maturity_bucket, row.maturity_bucket))}</b>",
            shown(row.pred),
            shown(row.bench_mean3),
            escape(status.get(row.status, str(row.status))),
            escape(ui.date_en(row.predicted_at.tz_convert("America/New_York"))),
        ]
        for row in waiting.itertuples()
    ]
    st.markdown(
        ui.panel(
            "Waiting for a result",
            ui.table(
                ["Date", "Auction", "Forecast", "Last 3", "Status", "Forecast made on"],
                rows,
                right={2, 3},
                scroll=True,
            ),  # fmt: skip
        ),
        unsafe_allow_html=True,
    )
