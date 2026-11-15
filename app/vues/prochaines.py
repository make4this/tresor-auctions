"""Page « Upcoming auctions » : toutes les prévisions des enchères à venir, et leur fiabilité."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from tresor import app_data as ad
from tresor import app_ui as ui
from tresor.config import SIGNAL_THRESHOLD
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


def display(frame: pd.DataFrame) -> pd.DataFrame:
    """Table affichée : l'essentiel d'abord (lisible sur téléphone)."""
    return pd.DataFrame(
        {
            "Date": [ui.date_en(day, year=False) for day in frame["auction_date"]],
            "Auction": frame["title"],
            "Expected demand": frame["signal"],
            "Gap": frame["gap_btc"].map(lambda v: en(v, 3, sign=True)),
            "Bid-to-cover forecast": frame["pred_btc"].map(lambda v: en(v, 2)),
            "Last-3 average": frame["bench_mean3_btc"].map(lambda v: en(v, 2)),
            "Size": [
                ad.amount(value, bool(estimated))
                for value, estimated in zip(
                    frame["offering_amt"], frame["offering_estimated"], strict=True
                )
            ],
            "Dealer share forecast": frame["pred_dealer_share"].map(
                lambda v: ad.format_target(v, "dealer_share")
            ),
            "Tail forecast": frame["pred_tail_bp"].map(lambda v: ad.format_target(v, "tail_bp")),
        }
    )


def colour(value: str) -> str:
    if value == ad.WEAK:
        return f"color: {ui.ALERT}; font-weight: 600"
    if value == ad.STRONG:
        return f"color: {ui.PRIMARY}; font-weight: 600"
    return ""


def show(frame: pd.DataFrame) -> None:
    shown = display(frame)
    st.dataframe(
        shown.style.map(colour, subset=["Expected demand"]),
        hide_index=True,
        width="stretch",
        height=35 * (len(shown) + 1) + 3,  # toutes les enchères, sans défilement
    )


if table.empty:
    st.info("No bill, note or bond auction has been announced yet.")
else:
    show(table)
    threshold = en(SIGNAL_THRESHOLD, 2)
    st.caption(
        f"Expected demand is **weaker** when the bid-to-cover forecast is at least {threshold}"
        f" below the average of the last 3 auctions of the same maturity, **stronger** when it is"
        f" at least {threshold} above, and **normal** otherwise. Size marked *est.*: not yet"
        " announced, so the size of the previous auction is used."
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
            show(fresh)

st.subheader("How reliable is this?")
record = ad.signal_track_record(ui.parquet("backtest_predictions.parquet")).set_index("family")
r2 = ad.backtest_r2(ui.csv("backtest_metrics.csv")).set_index(["family", "target"])
left, right = st.columns(2)
with left:
    st.markdown("**Past accuracy of the signal** (tests 2014-2026)")
    lines = []
    for family, label in ad.FAMILY_LABELS.items():
        if family not in record.index:
            continue
        row = record.loc[family]
        lines.append(
            f"- {label}: when the signal said *weaker*, the bid-to-cover did come in below the"
            f" last-3 average {en(row['weak_then_below'] * 100, 0)}% of the time (versus"
            f" {en(row['base_rate_below'] * 100, 0)}% for all auctions)."
        )
    st.markdown("\n".join(lines))
with right:
    st.markdown("**Forecast quality** (2020-2026, versus the last-3 average)")

    def value(family: str, target: str) -> str:
        row = r2.loc[(family, target)]
        return (
            f"{en(row['r2_vs_mean3'] * 100, 1, sign=True)}% [{en(row['r2_ci_low'] * 100, 0)},"
            f" {en(row['r2_ci_high'] * 100, 0)}]"
        )

    lines = []
    for family, label in ad.FAMILY_LABELS.items():
        parts = [
            f"{ad.SHORT_TARGETS[target]} {value(family, target)}"
            for target in ("btc", "dealer_share", "indirect_share")
            if (family, target) in r2.index
        ]
        lines.append(f"- {label}: " + ", ".join(parts) + ".")
    st.markdown("\n".join(lines))
st.caption(
    "Out-of-sample R² against the average of the last 3 auctions, with a 90% confidence"
    " interval: above 0, the forecast beats that simple rule. The bid-to-cover of notes and bonds"
    " is the most reliable result; for bills, part of the gain comes from the debt-ceiling"
    " episodes of 2015, 2021 and 2023. The signal describes expected demand, not where yields"
    " will go."
)
