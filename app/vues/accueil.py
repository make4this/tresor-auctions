"""Page d'accueil : tableau de bord en panneaux (enchères à venir, demande attendue, taux du
jour, historique du bid-to-cover, fiabilité des modèles, sources, résultats clés).

Tous les chiffres sont lus dans les fichiers de data/, comme sur les autres pages.
"""

from __future__ import annotations

from html import escape

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from tresor import app_data as ad
from tresor import app_ui as ui
from tresor.config import SIGNAL_THRESHOLD
from tresor.formats import en

ROWS = 10  # enchères affichées dans le panneau ; toutes sont sur la page « Upcoming auctions »
PERIODS = {"1Y": 1, "3Y": 3, "5Y": 5, "10Y": 10, "All": None}

st.markdown(
    '<p class="tr-intro">Forecasts of investor demand at each US Treasury auction, made the'
    " evening before with public data only. <b>Bid-to-cover</b> = amount bid ÷ amount sold:"
    " the higher, the stronger the demand.</p>",
    unsafe_allow_html=True,
)

log = ui.parquet("predictions_log.parquet")
auctions = ui.parquet("auctions.parquet", folder="clean")
upcoming = ad.upcoming_table(log, auctions, ui.today_new_york())
threshold = en(SIGNAL_THRESHOLD, 2)

# --- Deux colonnes : à gauche les enchères et le graphique, à droite les panneaux courts ------
# Conteneur nommé : sous 1 240 px de large, le CSS empile les deux colonnes (app_ui.PAGE_CSS).
left, right = st.container(key="home_grid").columns([1.8, 1], gap="medium")
with left:
    if upcoming.empty:
        body = (
            '<div class="tr-panel-note">No bill, note or bond auction has been announced yet.</div>'
        )
        note = ""
    else:
        rows = [
            [
                escape(ui.date_en(row.auction_date, year=False)),
                f"<b>{escape(row.title)}</b>",
                escape(ad.amount(row.offering_amt, bool(row.offering_estimated))),
                en(row.pred_btc, 2),
                en(row.bench_mean3_btc, 2),
                ui.signed(row.gap_btc, en(row.gap_btc, 2, sign=True)),
                ui.demand_tag(row.signal),
            ]
            for row in upcoming.head(ROWS).itertuples()
        ]
        headers = ["Date", "Auction", "Size", "Forecast", "Last 3", "Gap", "Demand"]
        body = ui.table(
            headers, rows, right={2, 3, 4, 5}, wrap={2}
        )  # « (est.) » à la ligne si besoin
        shown = f"Next {min(ROWS, len(upcoming))} of {len(upcoming)} announced auctions."
        note = (
            f"{shown} Forecast and Last 3: bid-to-cover predicted, and its average over the last 3"
            f" auctions of the same maturity. Demand is <b>weaker</b> when the gap is −{threshold}"
            f" or less, <b>stronger</b> when it is +{threshold} or more."
        )
    st.markdown(ui.panel("Upcoming auctions", body, note), unsafe_allow_html=True)

with right:
    counts = ad.demand_counts(upcoming)
    stats = "".join(
        f'<div class="tr-stat"><div class="tr-value">{counts[signal]}</div>'
        f'<div class="tr-text">{ui.demand_tag(signal)} demand</div></div>'
        for signal in (ad.WEAK, ad.NORMAL, ad.STRONG)
    )
    note = ""
    if not upcoming.empty:
        first = upcoming.iloc[0]
        note = (
            f"Next: {escape(ui.date_en(first['auction_date'], year=False))},"
            f" {escape(first['title'])} ({escape(ad.FAMILY_LABELS[first['family']].lower())})."
        )
    st.markdown(
        ui.panel("Demand outlook", f'<div class="tr-stats tr-three">{stats}</div>', note),
        unsafe_allow_html=True,
    )
    market = ad.market_snapshot(ui.parquet("market.parquet", folder="clean"))
    rows = [
        [
            escape(row.label),
            en(row.value, 2) + ("%" if row.unit == "bp" else ""),
            ui.signed(
                row.change, en(row.change, 0 if row.unit == "bp" else 2, sign=True) + f" {row.unit}"
            ),  # fmt: skip
        ]
        for row in market.itertuples()
    ]
    when = ui.day_en(market["date"].max()) if not market.empty else "–"
    st.markdown(
        ui.panel(
            "Market today",
            ui.table(["Series", "Level", "Change"], rows, right={1, 2}, wrap={0}),
            f"Close of {when}, change since the previous close.",
        ),
        unsafe_allow_html=True,
    )

# --- Suite des deux colonnes : historique du bid-to-cover | fiabilité et sources -------------
with left, st.container(border=True, key="panel_history"):
    st.markdown('<div class="tr-panel-head" style="padding: 0">Bid-to-cover history</div>',
                unsafe_allow_html=True)  # fmt: skip
    features = ui.parquet("features.parquet", folder="clean")
    period = st.segmented_control(
        "Period", list(PERIODS), default="5Y", required=True, label_visibility="collapsed"
    ) or "5Y"  # fmt: skip
    main = ["4W", "13W", "26W", "52W", "2Y", "5Y", "10Y", "30Y"]  # les plus suivies
    available = [b for b in main if b in set(features["maturity_bucket"])]
    buckets = st.pills(
        "Maturities", available, selection_mode="multi", default=["13W", "2Y", "10Y"],
        format_func=ad.MATURITY_LABELS.get, label_visibility="collapsed",
    ) or ["10Y"]  # fmt: skip
    buckets = buckets[: len(ui.SERIES)]  # 4 courbes au plus : une couleur chacune
    years = PERIODS[period]
    start = ui.today_new_york() - pd.DateOffset(years=years) if years else None
    history = ad.btc_history(features, buckets, start)
    figure = go.Figure()
    for color, bucket in zip(ui.SERIES, buckets, strict=False):
        part = history[history["maturity_bucket"] == bucket]
        if part.empty:
            continue
        label = ad.MATURITY_LABELS[bucket]
        figure.add_scatter(
            x=part["auction_date"], y=part["btc"], mode="lines", name=label,
            line={"color": color, "width": 1.6},
            hovertemplate=f"{label}<br>%{{x|%b %d, %Y}}<br>%{{y:.2f}}<extra></extra>",
        )  # fmt: skip
    ui.chart(figure, height=340)
    st.caption(
        "Bid-to-cover of each completed auction. Pick the maturities above; bills (weeks) usually"
        " attract more bids than notes and bonds (years)."
    )

with right:
    card = ad.scorecard(ui.csv("backtest_metrics.csv")).set_index(["family", "target"])
    cells = ["<div></div>"] + [
        f'<div class="tr-colhead">{escape(label)}</div>' for label in ad.FAMILY_LABELS.values()
    ]
    for target, label in ad.TARGET_LABELS.items():
        cells.append(f'<div class="tr-label">{escape(label)}</div>')
        for family in ad.FAMILY_LABELS:
            if (family, target) not in card.index:
                cells.append('<div class="tr-cell tr-unclear">–</div>')
                continue
            row = card.loc[(family, target)]
            value = en(row["r2_vs_mean3"] * 100, 0, sign=True) + "%"
            cells.append(f'<div class="tr-cell tr-{row["verdict"]}">{value}</div>')
    st.markdown(
        ui.panel(
            "Model scorecard",
            f'<div class="tr-grid">{"".join(cells)}</div>',
            "How much smaller the forecast errors are than with the last-3 average (out of"
            " sample, 2020-2026). Green: reliably better; red: reliably worse; grey: not clearly"
            " different. The tail mostly follows the day's market.",
        ),
        unsafe_allow_html=True,
    )
    quality = ad.quality_table(ui.json_file("data_quality.json"))
    colours = {"vert": ui.UP, "orange": "#D97706", "rouge": ui.ALERT}
    rows = [
        [
            f'<span class="tr-dot" style="background:{colours.get(row.status, ui.MUTED)}">'
            f"</span>{escape(ad.SOURCE_LABELS.get(row.source, row.source))}",
            escape(ui.day_en(row.last_date, year=False)),
        ]
        for row in quality.itertuples()
    ]
    st.markdown(
        ui.panel(
            "Data sources",
            ui.table(["Source", "Latest data"], rows, right={1}, wrap={0}),
            "Green: up to date. Details on the Data quality page.",
        ),
        unsafe_allow_html=True,
    )

# --- Ligne 3 : résultats clés ----------------------------------------------------------------
facts = ad.key_facts(
    ui.csv("backtest_metrics.csv"), ui.csv("strategy_summary.csv"), ui.csv("quintiles.csv")
)
# R² contre mean3 = baisse de l'erreur quadratique : écrite comme une variation de l'erreur.
findings = [
    (
        en(-facts["r2_coupons"] * 100, 0, sign=True, suffix="%"),
        'squared error versus the simple rule "same as the last 3 auctions" for notes and'
        f" bonds ({en(-facts['r2_bills'] * 100, 0, sign=True)}% for bills), 2020-2026.",
    ),
    (
        f"{en(facts['weak_move_bp'], 1, sign=True)} bp",
        "average yield move on auction day for the 20% of note and bond auctions with the"
        " weakest demand relative to usual, 2009-2026.",
    ),
    (
        f"{en(facts['strategy_bp'], 1, sign=True)} bp",
        "per trade after costs for a strategy that follows the signal, 2020-2026: forecasting"
        " demand is not enough to make money.",
    ),
]
stats = "".join(
    f'<div class="tr-stat"><div class="tr-value">{escape(value)}</div>'
    f'<div class="tr-text">{escape(text)}</div></div>'
    for value, text in findings
)
st.markdown(
    ui.panel("Key findings", f'<div class="tr-stats">{stats}</div>'),
    unsafe_allow_html=True,
)
