"""Page « Dataset » : la base du site, téléchargeable en CSV, avec le sens de chaque colonne."""

from __future__ import annotations

from html import escape

import pandas as pd
import streamlit as st

from tresor import app_data as ad
from tresor import app_ui as ui
from tresor.formats import en

ui.page_header(
    "Dataset",
    "The site's own database, rebuilt every weekday evening from public sources and free to"
    " download. Every predictor carries only information published before the auction it"
    " describes.",
)

DATASETS = [
    (
        "auctions",
        "Treasury auctions since 2000",
        "auctions.parquet",
        "clean",
        "treasury_auctions.csv",
        ad.AUCTION_COLUMNS,
        "Every bill, note, bond, TIPS, FRN and cash management bill auction since 2000, cleaned:"
        " auctions found in both Fiscal Data and TreasuryDirect are merged, and the announced"
        " auctions are included.",
    ),
    (
        "model",
        "Model dataset (point-in-time)",
        "features.parquet",
        "clean",
        "model_dataset_point_in_time.csv",
        ad.MODEL_ID_COLUMNS,
        "One row per bill, note and bond auction since 2000: the predictors used by the models,"
        " each built only from values published strictly before the auction day, and the 4"
        " measures to forecast. The models learn from 2009 on; earlier rows feed the"
        " last-auction averages.",
    ),
    (
        "forecasts",
        "Forecast log",
        "live_tracking.parquet",
        "output",
        "forecast_log.csv",
        ad.FORECAST_COLUMNS,
        "Every live forecast, frozen on auction day and never changed afterwards, with the"
        " published result once it is known.",
    ),
]

for key, title, name, folder, filename, meanings, text in DATASETS:
    frame = ui.parquet(name, folder=folder)
    public = ad.public_table(frame)
    dates = pd.to_datetime(frame["auction_date"])
    first, last = (f"{dates.min():%Y}", f"{dates.max():%Y}") if not frame.empty else ("–", "–")
    period = first if first == last else f"{first}–{last}"
    data = ui.public_csv(name, folder=folder)
    with ui.box(f"dataset_{key}", title):
        st.markdown(
            f'<div class="tr-small" style="margin-bottom:0.4rem">{escape(text)}</div>'
            + ui.stats(
                [
                    (en(len(public), 0), "rows"),
                    (en(public.shape[1], 0), "columns"),
                    (period, "auction dates"),
                    (en(len(data) / 1e6, 1, suffix=" MB"), "CSV size"),
                ]
            ),
            unsafe_allow_html=True,
        )
        st.download_button(
            f"Download {filename}",
            data=data,
            file_name=filename,
            mime="text/csv",
            icon=":material/download:",
            key=f"download_{key}",
        )
        with st.expander("What each column means"):
            guide = ad.column_guide(frame, meanings)
            st.markdown(
                ui.table(
                    ["Column", "Meaning"],
                    [
                        [f"<code>{escape(c)}</code>", escape(m)]
                        for c, m in guide.itertuples(index=False)
                    ],
                    wrap={1},
                    scroll=True,
                ),
                unsafe_allow_html=True,
            )

restricted = ", ".join(sorted({f"{p}" for p in ad.RESTRICTED_COLUMNS.values()}))
st.markdown(
    ui.panel(
        "Sources and reuse",
        '<div class="tr-body"><ul>'
        "<li>Auction data: US Treasury (Fiscal Data, TreasuryDirect). Market and macro data:"
        " Treasury.gov, FRED (St. Louis Fed), New York Fed, CFTC.</li>"
        f"<li>Columns based on data from {escape(restricted)} (the VIX and the Baa credit"
        " spread) are used by the models but left out of the downloads, because their providers"
        " restrict redistribution.</li>"
        "<li>Dates are New York dates; amounts are in dollars, rates in %, the tail in basis"
        " points.</li></ul></div>",
    ),
    unsafe_allow_html=True,
)
