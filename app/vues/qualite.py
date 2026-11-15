"""Page « Data quality » : fraîcheur de chaque source, contrôles et rapport de nettoyage."""

from __future__ import annotations

from html import escape

import pandas as pd
import streamlit as st

from tresor import app_data as ad
from tresor import app_ui as ui
from tresor.clean.checks import MAX_DELAY_BUSINESS_DAYS, MAX_DELAY_OVERRIDES
from tresor.formats import en
from tresor.sources.collect import FETCH_STATUS_FILE

ui.page_header(
    "Data quality",
    "Every evening the site downloads its public sources, cleans them and checks them before"
    " forecasting. This page shows how fresh each source is and what the checks found.",
)
quality = ui.json_file("data_quality.json")
report = ui.json_file("cleaning_report.json")
fetch = ui.json_file(FETCH_STATUS_FILE)
if not quality:
    st.error("Quality report missing: run `python -m tresor.pipeline --clean`.")
    st.stop()

STATUS = {"vert": ("Good", ui.UP), "orange": ("Watch", "#D97706"), "rouge": ("Problem", ui.ALERT)}


def status_html(value: object) -> str:
    """Point de couleur et mot : Good, Watch ou Problem."""
    label, colour = STATUS.get(str(value), (str(value), ui.MUTED))
    return f'<span class="tr-dot" style="background:{colour}"></span>{escape(label)}'


def count(value: object) -> int:
    """Nombre de cas d'un contrôle, qu'il soit écrit comme un nombre ou comme une liste."""
    if isinstance(value, list | dict):
        return len(value)
    try:
        return int(value)  # type: ignore[call-overload]
    except (TypeError, ValueError):
        return 0


auctions = quality.get("auctions", {})
market = quality.get("market", {})
auctions = auctions if isinstance(auctions, dict) else {}
market = market if isinstance(market, dict) else {}
checks = [
    ("Auctions: duplicate keys", auctions.get("duplicate_keys")),
    ("Recent auctions without a result", auctions.get("recent_without_results")),
    ("Regular series interrupted", auctions.get("stale_regular_series")),
    ("Market: duplicate keys", market.get("duplicate_keys")),
    ("Market: rates out of bounds", market.get("rates_outside_bounds")),
    ("Market: values without a publication date",
     market.get("values_without_valid_publication_date")),
]  # fmt: skip
table = ad.quality_table(quality)
fresh = int((table["status"] == "vert").sum()) if not table.empty else 0
flagged = sum(1 for _, value in checks if count(value))
st.markdown(
    ui.panel(
        "Overview",
        ui.stats(
            [
                (status_html(quality.get("status")), "overall status"),
                (f"{fresh} of {len(table)}", "sources up to date"),
                (str(flagged), "checks with cases found"),
            ],
            three=True,
        ),
        f"Checks of {escape(ui.day_en(quality.get('today_new_york')))} (New York date).",
        [
            "<b>Good</b>: nothing to report. <b>Watch</b>: extreme values to verify."
            " <b>Problem</b>: an alert, such as a late source, missing rows or a failed check.",
            "If a source fails, the forecasts use its latest saved data.",
        ],
    ),
    unsafe_allow_html=True,
)
failed = ad.failed_sources(fetch)
if failed:
    st.warning(
        f"Last download, {ui.moment_en(fetch.get('run_at'))}: {len(failed)} source(s) failed"
        f" ({', '.join(failed)}). Forecasts use their latest saved data; how fresh it is is"
        " checked in the table below."
    )

overrides = ", ".join(
    f"{en(days, 0)} for {ad.SOURCE_LABELS.get(name, name)}"
    for name, days in MAX_DELAY_OVERRIDES.items()
)
rows = [
    [
        escape(ad.SOURCE_LABELS.get(item.source, item.source)),
        status_html(item.status),
        en(item.rows, 0),
        escape(ui.day_en(item.last_date)),
        escape(ui.day_en(item.last_published)),
        en(item.business_days_since_last, 0),
        escape(str(item.alerts)),
    ]
    for item in table.itertuples()
]
st.markdown(
    ui.panel(
        "Sources",
        ui.table(
            [
                "Source",
                "Status",
                "Rows",
                "Latest value",
                "Published",
                "Days since",
                "Alerts",
            ],
            rows,
            right={2, 5},
            wrap={0, 5, 6},
        ),  # fmt: skip
        f"A source is flagged when its latest publication is more than {MAX_DELAY_BUSINESS_DAYS}"
        f" business days old ({escape(overrides)}, which TreasuryDirect completes), or when its"
        " number of rows drops abnormally.",
        [
            "<b>Latest value</b>: the date the newest value refers to. <b>Published</b>: the"
            " date it became public. <b>Days since</b>: business days since that publication.",
            "Only values published before an auction are used to forecast it.",
            "The auction calendar has future dates, hence a negative delay.",
        ],
    ),
    unsafe_allow_html=True,
)

downloads = ad.fetch_table(fetch)
if downloads.empty:
    body = (
        '<div class="tr-panel-note">Download report missing: it is written by'
        " <code>--fetch</code>.</div>"
    )
    note = ""
else:
    results = {
        True: '<span class="tr-tag tr-strong">Received</span>',
        False: '<span class="tr-tag tr-weak">Failed</span>',
    }
    rows = [
        [
            escape(ad.SOURCE_LABELS.get(item.source, item.source)),
            results[bool(item.ok)],
            escape(ui.moment_en(item.last_success_at)),
            escape(ad.error_en(str(item.error))),
        ]
        for item in downloads.itertuples()
    ]
    body = ui.table(["Source", "Result", "Last success", "Error"], rows, wrap={0, 3})
    note = f"Run of {escape(ui.moment_en(fetch.get('run_at')))}."
st.markdown(ui.panel("Last download", body, note), unsafe_allow_html=True)

rows = [[escape(name), en(count(value), 0)] for name, value in checks]
st.markdown(
    ui.panel(
        "Checks",
        ui.table(["Check", "Cases found"], rows, right={1}, wrap={0}),
        guide=[
            "Automatic checks run after cleaning; 0 everywhere is the normal state.",
            "<b>Values without a publication date</b>: a value whose release date is unknown"
            " cannot be used, since the site only forecasts with what was public.",
        ],
    ),
    unsafe_allow_html=True,
)
for level, items in (
    ("Alerts", quality.get("alerts", [])),
    ("Warnings", quality.get("warnings", [])),
):
    if items:
        st.warning(f"{level}: " + "; ".join(map(str, items)))

cleaning = report.get("auctions", {}) if isinstance(report, dict) else {}
if cleaning:
    rows_in = cleaning.get("rows_in", {})
    excluded = cleaning.get("excluded", {})
    merged = int(cleaning.get("rows_after_merge", 0))
    dropped = sum(int(info.get("rows", 0)) for info in excluded.values())
    received = ", ".join(f"{ad.INPUT_LABELS.get(n, n)} {en(c, 0)}" for n, c in rows_in.items())
    body = ui.stats(
        [
            (en(cleaning.get("duplicates_removed", 0), 0), "auctions found in several sources,"
             " merged"),
            (en(merged, 0), "auctions after merging"),
            (en(merged - dropped, 0), "auctions kept after exclusions"),
        ],
        three=True,
    )  # fmt: skip
    if excluded:
        body += ui.table(
            ["Exclusion", "Rows"],
            [
                [escape(ad.EXCLUSION_LABELS.get(name, name)), en(info.get("rows", 0), 0)]
                for name, info in excluded.items()
            ],
            right={1},
            wrap={0},
        )
    st.markdown(
        ui.panel("Auction cleaning", body, f"Rows received: {escape(received)}."),
        unsafe_allow_html=True,
    )
    if excluded:
        with st.expander("Examples of excluded rows"):
            for name, info in excluded.items():
                st.markdown(f"**{ad.EXCLUSION_LABELS.get(name, name)}**")
                examples = pd.DataFrame(info.get("examples", []))
                if examples.empty:
                    continue
                shown = pd.DataFrame(
                    {
                        "CUSIP": examples.get("cusip"),
                        "Date": examples.get("auction_date", pd.Series(dtype=str)).map(ui.day_en),
                        "Type": examples.get("security_type"),
                        "Term": examples.get("security_term"),
                        "Size": examples.get("offering_amt", pd.Series(dtype=float)).map(
                            lambda v: f"${en(v / 1e6, 0)}m"
                        ),
                    }
                )
                st.dataframe(shown, hide_index=True, width="stretch")
