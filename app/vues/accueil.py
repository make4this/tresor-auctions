"""Page d'accueil : tableau de bord en panneaux (enchères à venir, lecture d'une enchère,
demande attendue, taux du jour, fiabilité des modèles, sources, résultats clés).

Chaque panneau a une courte aide « How to read », ouverte au clic sur son titre. Tous les
chiffres sont lus dans les fichiers de data/, comme sur les autres pages.
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
PERIODS = {"6M": 6, "1Y": 12, "3Y": 36, "All": None}  # en mois
# Vue par défaut : 6 mois pour les bills (une enchère par semaine), 3 ans pour les coupons.
DEFAULT_PERIOD = {"bill": "6M", "nominal_coupon": "3Y"}
THRESHOLD = en(SIGNAL_THRESHOLD, 2)
FORECAST_COLOURS = {ad.WEAK: ui.ALERT, ad.STRONG: ui.UP, ad.NORMAL: "#64748B"}

GUIDES = {
    "upcoming": ui.upcoming_guide(),
    "focus": [
        "<b>Blue line</b>: bid-to-cover of each past auction of this maturity; the 3 dots are"
        " the last 3.",
        "<b>Dashed line</b>: their average, the usual level the forecast is compared with.",
        f"<b>Diamond</b>: the forecast. Red if at least {THRESHOLD} below the dashed line"
        f" (weaker demand), green if at least {THRESHOLD} above (stronger), grey otherwise.",
        "Bills (weeks) usually draw more bids than notes and bonds (years): compare an auction"
        " with its own history, not with other maturities.",
    ],
    "market": [
        "Yields in % at the latest close; change since the previous close in bp"
        " (1 bp = 0.01 percentage point).",
        "<b>VIX</b>: expected volatility of US stocks over 30 days, a gauge of market stress.",
        "Both feed the models: auctions are priced against these yields.",
    ],
    "scorecard": [
        "Each cell: how much smaller the squared forecast error is than with the last-3"
        " average, out of sample over 2020-2026 (+10% = errors 10% smaller).",
        "<b>Green</b>: better, with 90% confidence. <b>Red</b>: worse. <b>Grey</b>: no clear"
        " difference.",
        "The tail mostly follows the market move of the day, which no model sees in advance.",
    ],
    "sources": [
        "<b>Green</b>: up to date. <b>Red</b>: an alert, such as a late source or missing rows.",
        "<b>Latest data</b>: date of the most recent value received.",
        "If a source fails, the forecasts use its latest saved data.",
    ],
}

st.markdown(
    '<p class="tr-intro">Forecasts of investor demand at each US Treasury auction, made the'
    " evening before with public data only. <b>Bid-to-cover</b> = amount bid ÷ amount sold:"
    " the higher, the stronger the demand. Start with the table, pick an auction below to see"
    " why, and click a panel title for help.</p>",
    unsafe_allow_html=True,
)

log = ui.parquet("predictions_log.parquet")
auctions = ui.parquet("auctions.parquet", folder="clean")
upcoming = ad.upcoming_table(log, auctions, ui.today_new_york())

# --- Deux colonnes : à gauche les enchères, à droite les panneaux courts ---------------------
# Conteneur nommé : sous 1 240 px de large, le CSS empile les deux colonnes (app_ui.PAGE_CSS).
left, right = st.container(key="home_grid").columns([1.8, 1], gap="medium")
with left:
    if upcoming.empty:
        body = (
            '<div class="tr-panel-note">No bill, note or bond auction has been announced yet.</div>'
        )
        note = ""
    else:
        body = ui.upcoming_html(upcoming.head(ROWS))
        note = (
            f"Next {min(ROWS, len(upcoming))} of {len(upcoming)} announced auctions. All of them,"
            " with the other measures, are on the Upcoming auctions page."
        )
    st.markdown(
        ui.panel("Upcoming auctions", body, note, GUIDES["upcoming"]), unsafe_allow_html=True
    )

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
            ui.signed_number(row.change, 0 if row.unit == "bp" else 2, f" {row.unit}"),
        ]
        for row in market.itertuples()
    ]
    when = ui.day_en(market["date"].max()) if not market.empty else "–"
    st.markdown(
        ui.panel(
            "Market today",
            ui.table(["Series", "Level", "Change"], rows, right={1, 2}, wrap={0}),
            f"Close of {when}.",
            GUIDES["market"],
        ),
        unsafe_allow_html=True,
    )


def focus_chart(row: pd.Series, period: str) -> go.Figure:
    """Bid-to-cover passé de la maturité, moyenne des 3 dernières et prévision (losange)."""
    features = ui.parquet("features.parquet", folder="clean")
    past = ad.btc_history(features, [row["maturity_bucket"]])
    past = past[past["auction_date"] < row["auction_date"]]
    last3 = past.tail(3)
    months = PERIODS[period]
    if months:
        past = past[past["auction_date"] >= row["auction_date"] - pd.DateOffset(months=months)]
    label = ad.MATURITY_LABELS.get(row["maturity_bucket"], row["maturity_bucket"])
    hover = "%{x|%b %d, %Y}<br>Bid-to-cover %{y:.2f}"
    figure = go.Figure()
    figure.add_scatter(
        x=past["auction_date"], y=past["btc"], mode="lines", name=f"Past {label} auctions",
        line={"color": ui.SERIES[0], "width": 2}, hovertemplate=hover + "<extra></extra>",
    )  # fmt: skip
    figure.add_scatter(
        x=last3["auction_date"], y=last3["btc"], mode="markers", showlegend=False,
        marker={"color": ui.SERIES[0], "size": 9, "line": {"color": "#FFFFFF", "width": 2}},
        hovertemplate=hover + "<extra>one of the last 3</extra>",
    )  # fmt: skip
    mean3, pred = row["bench_mean3_btc"], row["pred_btc"]
    if not last3.empty and pd.notna(mean3):
        figure.add_scatter(
            x=[last3["auction_date"].min(), row["auction_date"]], y=[mean3, mean3],
            mode="lines", name=f"Last-3 average {en(mean3, 2)}",
            line={"color": ui.NAVY, "dash": "dash", "width": 2},
            hovertemplate=f"Last-3 average {en(mean3, 2)}<extra></extra>",
        )  # fmt: skip
    if pd.notna(pred):
        figure.add_scatter(
            x=[row["auction_date"]], y=[pred], mode="markers", name=f"Forecast {en(pred, 2)}",
            marker={"symbol": "diamond", "size": 14, "line": {"color": "#FFFFFF", "width": 2},
                    "color": FORECAST_COLOURS.get(row["signal"], "#64748B")},
            hovertemplate=f"{ui.date_en(row['auction_date'])}<br>Forecast {en(pred, 2)}"
            "<extra></extra>",
        )  # fmt: skip
    first = past["auction_date"].min() if not past.empty else row["auction_date"]
    room = pd.Timedelta(days=max(3, (row["auction_date"] - first).days // 25))  # marge à droite
    figure.update_xaxes(range=[first - room, row["auction_date"] + room])
    return figure


# --- Suite des deux colonnes : lecture d'une enchère | fiabilité et sources ------------------
with left, st.container(border=True, key="panel_focus"):
    st.markdown(ui.panel_head("Auction in detail", GUIDES["focus"]), unsafe_allow_html=True)
    if upcoming.empty:
        st.markdown(
            '<div class="tr-small">The detail appears here once an auction is announced.</div>',
            unsafe_allow_html=True,
        )
    else:
        names = [
            " · ".join([ui.date_en(r.auction_date, year=False), r.title,
                        ad.amount(r.offering_amt, bool(r.offering_estimated))])
            for r in upcoming.itertuples()
        ]  # fmt: skip
        choice = st.selectbox(
            "Auction", range(len(upcoming)), format_func=names.__getitem__,
            label_visibility="collapsed",
        )  # fmt: skip
        row = upcoming.iloc[choice]
        st.markdown(
            f'<div class="tr-reading">{ui.demand_tag(row["signal"])}'
            f"{escape(ad.auction_reading(row))}</div>",
            unsafe_allow_html=True,
        )
        default = DEFAULT_PERIOD.get(row["family"], "1Y")
        period = st.segmented_control(
            "Period", list(PERIODS), default=default, required=True,
            label_visibility="collapsed", key=f"focus_period_{row['family']}",
        ) or default  # fmt: skip
        ui.chart(focus_chart(row, period), height=320)
        record = ad.signal_track_record(ui.parquet("backtest_predictions.parquet"))
        sentence = ad.signal_record_sentence(record, row["family"], row["signal"])
        if sentence:
            st.markdown(f'<div class="tr-small">{escape(sentence)}</div>', unsafe_allow_html=True)

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
            score = card.loc[(family, target)]
            value = en(score["r2_vs_mean3"] * 100, 0, sign=True) + "%"
            cells.append(f'<div class="tr-cell tr-{score["verdict"]}">{value}</div>')
    st.markdown(
        ui.panel(
            "Model scorecard",
            f'<div class="tr-grid">{"".join(cells)}</div>',
            "Forecast errors versus the last-3 average, 2020-2026.",
            GUIDES["scorecard"],
        ),
        unsafe_allow_html=True,
    )
    quality = ad.quality_table(ui.json_file("data_quality.json"))
    colours = {"vert": ui.UP, "orange": "#D97706", "rouge": ui.ALERT}
    rows = [
        [
            f'<span class="tr-dot" style="background:{colours.get(item.status, ui.MUTED)}">'
            f"</span>{escape(ad.SOURCE_LABELS.get(item.source, item.source))}",
            escape(ui.day_en(item.last_date, year=False)),
        ]
        for item in quality.itertuples()
    ]
    st.markdown(
        ui.panel(
            "Data sources",
            ui.table(["Source", "Latest data"], rows, right={1}, wrap={0}),
            "Details on the Data quality page.",
            GUIDES["sources"],
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
