"""Page « Upcoming auctions » : toutes les prévisions des enchères à venir, et le bilan passé
du signal."""

from __future__ import annotations

from html import escape

import pandas as pd
import streamlit as st

from tresor import app_data as ad
from tresor import app_ui as ui
from tresor.formats import en

st.title("Upcoming auctions")
st.caption(
    "Forecasts of the 5-model average for every announced auction. A forecast is frozen on"
    " auction day and never changed afterwards."
)

log = ui.parquet("predictions_log.parquet")
auctions = ui.parquet("auctions.parquet", folder="clean")
today = ui.today_new_york()
table = ad.upcoming_table(log, auctions, today)

if table.empty:
    body = '<div class="tr-panel-note">No bill, note or bond auction has been announced yet.</div>'
else:
    body = ui.upcoming_html(table, more=True)
st.markdown(
    ui.panel("All announced auctions", body, guide=ui.upcoming_guide(more=True)),
    unsafe_allow_html=True,
)

provisional = st.session_state.get("provisional")
if provisional is not None and not provisional.empty:
    computed = pd.Timestamp(provisional["computed_at"].iloc[0])
    with st.expander(f"Provisional forecast of {ui.moment_en(computed)} (not recorded)"):
        fresh = ad.upcoming_table(
            provisional.assign(predicted_at=provisional["computed_at"]), auctions, today
        )
        if fresh.empty:
            st.write("No upcoming auction in this refresh.")
        else:
            st.markdown(ui.upcoming_html(fresh, more=True), unsafe_allow_html=True)

record = ad.signal_record_table(ad.signal_track_record(ui.parquet("backtest_predictions.parquet")))
rows = [
    [
        escape(ad.FAMILY_LABELS.get(item.family, item.family)),
        ui.demand_tag(item.signal),
        en(item.cases, 0),
        f"{en(item.right, 0)} ({en(item.share * 100, 0)}%)" if item.cases else "–",
        f"{en(item.base * 100, 0)}%",
    ]
    for item in record.itertuples()
]
st.markdown(
    ui.panel(
        "Track record of the signal",
        ui.table(
            ["Securities", "Signal", "Times given", "Came true", "All auctions"],
            rows,
            right={2, 3, 4},
        ),
        "Out-of-sample tests 2014-2026, 5-model average, bid-to-cover. How accurate each measure"
        " is: Model scorecard on Home and the Model performance page.",
        [
            "<b>Came true</b>: the bid-to-cover landed on the announced side of the last-3"
            " average (below for <i>weaker</i>, above for <i>stronger</i>), by any amount.",
            "<b>All auctions</b>: how often that happens for any auction, i.e. without the"
            " signal. The gap between the two columns is what the signal adds.",
            "The signal describes expected demand at the auction, not where yields will go.",
        ],
    ),
    unsafe_allow_html=True,
)
