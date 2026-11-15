"""Page « Data quality » : fraîcheur de chaque source, contrôles et rapport de nettoyage."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from tresor import app_data as ad
from tresor import app_ui as ui
from tresor.clean.checks import MAX_DELAY_BUSINESS_DAYS, MAX_DELAY_OVERRIDES
from tresor.formats import en
from tresor.sources.collect import FETCH_STATUS_FILE

st.title("Data quality")
quality = ui.json_file("data_quality.json")
report = ui.json_file("cleaning_report.json")
fetch = ui.json_file(FETCH_STATUS_FILE)
if not quality:
    st.error("Quality report missing: run `python -m tresor.pipeline --clean`.")
    st.stop()

STATUS = {"vert": "🟢 good", "orange": "🟠 watch", "rouge": "🔴 problem"}


def count(value: object) -> int:
    """Nombre de cas d'un contrôle, qu'il soit écrit comme un nombre ou comme une liste."""
    if isinstance(value, list | dict):
        return len(value)
    try:
        return int(value)  # type: ignore[call-overload]
    except (TypeError, ValueError):
        return 0


st.markdown(
    f"**Overall status: {STATUS.get(str(quality.get('status')), '?')}**, checks of"
    f" {ui.day_en(quality.get('today_new_york'))} (New York date)."
)
failed = ad.failed_sources(fetch)
if failed:
    st.warning(
        f"Last download, {ui.moment_en(fetch.get('run_at'))}: {len(failed)} source(s) failed"
        f" ({', '.join(failed)}). Forecasts use their latest saved data; how fresh it is is"
        " checked in the table below."
    )

table = ad.quality_table(quality)
st.subheader("Sources")
if not table.empty:
    st.dataframe(
        pd.DataFrame(
            {
                "Source": table["source"].map(lambda s: ad.SOURCE_LABELS.get(s, s)),
                "Status": table["status"].map(lambda s: STATUS.get(str(s), str(s))),
                "Rows": table["rows"].map(lambda v: en(v, 0)),
                "Latest observation": table["last_date"].map(ui.day_en),
                "Latest publication": table["last_published"].map(ui.day_en),
                "Business days since publication": table["business_days_since_last"].map(
                    lambda v: en(v, 0)
                ),
                "Alerts": table["alerts"],
            }
        ),
        hide_index=True,
        width="stretch",
        height=35 * (len(table) + 1) + 3,  # toutes les sources, sans défilement
    )
overrides = ", ".join(
    f"{en(days, 0)} for {ad.SOURCE_LABELS.get(name, name)}"
    for name, days in MAX_DELAY_OVERRIDES.items()
)
st.caption(
    f"A source is flagged when its latest publication is more than {MAX_DELAY_BUSINESS_DAYS}"
    f" business days old ({overrides}, which TreasuryDirect completes), or when its number of"
    " rows drops abnormally. The auction calendar has future dates, hence a negative delay."
)

st.subheader("Last download")
downloads = ad.fetch_table(fetch)
if downloads.empty:
    st.caption("Download report missing: it is written by `--fetch`.")
else:
    st.caption(f"Run of {ui.moment_en(fetch.get('run_at'))}.")
    st.dataframe(
        pd.DataFrame(
            {
                "Source": downloads["source"].map(lambda s: ad.SOURCE_LABELS.get(s, s)),
                "Result": downloads["ok"].map({True: "🟢 received", False: "🔴 failed"}),
                "Last success": downloads["last_success_at"].map(ui.moment_en),
                "Error": downloads["error"],
            }
        ),
        hide_index=True,
        width="stretch",
        height=35 * (len(downloads) + 1) + 3,
    )

auctions = quality.get("auctions", {})
market = quality.get("market", {})
auctions = auctions if isinstance(auctions, dict) else {}
market = market if isinstance(market, dict) else {}
st.subheader("Checks")
checks = [
    ("Auctions: duplicate keys", auctions.get("duplicate_keys")),
    ("Recent auctions without a result", auctions.get("recent_without_results")),
    ("Regular series interrupted", auctions.get("stale_regular_series")),
    ("Market: duplicate keys", market.get("duplicate_keys")),
    ("Market: rates out of bounds", market.get("rates_outside_bounds")),
    (
        "Market: values without a publication date",
        market.get("values_without_valid_publication_date"),
    ),
]
st.dataframe(
    pd.DataFrame({"Check": [c for c, _ in checks], "Cases found": [count(v) for _, v in checks]}),
    hide_index=True,
    width="stretch",
)
for level, items in (
    ("Alerts", quality.get("alerts", [])),
    ("Warnings", quality.get("warnings", [])),
):
    if items:
        st.warning(f"{level}: " + "; ".join(map(str, items)))

cleaning = report.get("auctions", {}) if isinstance(report, dict) else {}
if cleaning:
    st.subheader("Auction cleaning")
    rows_in = cleaning.get("rows_in", {})
    excluded = cleaning.get("excluded", {})
    merged = int(cleaning.get("rows_after_merge", 0))
    dropped = sum(int(info.get("rows", 0)) for info in excluded.values())
    st.markdown(
        "Rows received: "
        + ", ".join(f"{ad.INPUT_LABELS.get(n, n)} {en(c, 0)}" for n, c in rows_in.items())
        + ". Auctions found in several sources (merged):"
        f" {en(cleaning.get('duplicates_removed', 0), 0)}. After merging: {en(merged, 0)};"
        f" after exclusions: **{en(merged - dropped, 0)} auctions**."
    )
    if excluded:
        st.dataframe(
            pd.DataFrame(
                {
                    "Exclusion": [ad.EXCLUSION_LABELS.get(name, name) for name in excluded],
                    "Rows": [info.get("rows", 0) for info in excluded.values()],
                }
            ),
            hide_index=True,
            width="stretch",
        )
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
