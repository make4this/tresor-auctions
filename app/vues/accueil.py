"""Page d'accueil : le projet et les prochaines enchères, en mots simples (en anglais).

Les détails (tableaux complets, backtest, qualité des données) sont dans les pages « More
details ». Tous les chiffres sont lus dans les fichiers de data/, comme sur les autres pages.
"""

from __future__ import annotations

from html import escape

import streamlit as st

from tresor import app_data as ad
from tresor import app_ui as ui
from tresor.config import SIGNAL_THRESHOLD
from tresor.formats import en

st.markdown(
    """
<div class="tr-hero">
<h1>US Treasury auctions,<br><span class="tr-soft">forecast before they happen</span></h1>
<p>Every week, the US government borrows tens of billions of dollars by auctioning Treasury
securities. This site estimates, before each auction, whether investor demand will be weaker or
stronger than usual.</p>
</div>
""",
    unsafe_allow_html=True,
)

quality = ui.json_file("data_quality.json")
tables = len(quality["sources"]) if isinstance(quality.get("sources"), dict) else 11
st.markdown(
    f"""
<div class="tr-steps">
<div class="tr-step"><span class="tr-number">1</span><h4>Collect</h4>
<p>Every evening, {tables} free public datasets (US Treasury, Federal Reserve, CFTC, Cboe) are
downloaded automatically.</p></div>
<div class="tr-step"><span class="tr-number">2</span><h4>Forecast</h4>
<p>For each announced auction, 5 models estimate demand using only what was known the day
before. The forecast is their average.</p></div>
<div class="tr-step"><span class="tr-number">3</span><h4>Check</h4>
<p>Each forecast is frozen on auction day, then compared with the actual result, published that
same afternoon.</p></div>
</div>
""",
    unsafe_allow_html=True,
)

st.subheader("Upcoming auctions")
log = ui.parquet("predictions_log.parquet")
auctions = ui.parquet("auctions.parquet", folder="clean")
cards = ad.home_cards(ad.upcoming_table(log, auctions, ui.today_new_york()))
FIRST_CARDS = 6  # les suivantes sont repliées : la page reste courte


def show_cards(selection: list[dict[str, str]]) -> None:
    blocks = [
        f"""<div class="tr-card"><span class="tr-date">{escape(ui.date_en(card["date"]))} ·
{escape(card["family"])} · {escape(card["amount"])}</span><h4>{escape(card["title"])}</h4>
<span class="tr-badge {card["css"]}">{escape(card["badge"])}</span>
<p>{escape(card["sentence"])}</p></div>"""
        for card in selection
    ]
    st.markdown(f'<div class="tr-cards">{"".join(blocks)}</div>', unsafe_allow_html=True)


if not cards:
    st.info("No bill, note or bond auction has been announced yet.")
else:
    show_cards(cards[:FIRST_CARDS])
    if len(cards) > FIRST_CARDS:
        with st.expander(f"Show {len(cards) - FIRST_CARDS} more announced auctions"):
            show_cards(cards[FIRST_CARDS:])

threshold = en(SIGNAL_THRESHOLD, 2)
st.markdown(
    f"""
<div class="tr-panel">
<h4>How to read a card</h4>
<p>The <b>bid-to-cover ratio</b> is the amount investors bid divided by the amount sold: 2.8
means they asked for 2.8 times what the Treasury was selling. The higher it is, the stronger the
demand.</p>
<p>Demand is expected to be <b>weaker</b> when the forecast is at least {threshold} below the
average of the last 3 auctions of the same maturity, <b>stronger</b> when it is at least
{threshold} above, and <b>normal</b> otherwise.</p>
</div>
""",
    unsafe_allow_html=True,
)

st.subheader("What the tests since 2014 show")
facts = ad.key_facts(
    ui.csv("backtest_metrics.csv"), ui.csv("strategy_summary.csv"), ui.csv("quintiles.csv")
)
# R² contre mean3 = baisse de l'erreur quadratique : écrite comme une variation de l'erreur.
coupons = en(-facts["r2_coupons"] * 100, 0, sign=True, suffix="%")
bills = en(-facts["r2_bills"] * 100, 0, sign=True, suffix="%")
st.markdown(
    f"""
<div class="tr-facts">
<div class="tr-fact"><div class="tr-value">{coupons}</div>
<p>squared error compared with the simple rule "same as the last 3 auctions", for notes and
bonds ({bills} for bills, 2020-2026). The forecast helps, a little.</p></div>
<div class="tr-fact"><div class="tr-value">{en(facts["weak_move_bp"], 1, sign=True)} bp</div>
<p>average rise in the yield on auction day (1 bp = 0.01 percentage point) for the 20% of note
and bond auctions with the weakest demand relative to usual (2009-2026).</p></div>
<div class="tr-fact"><div class="tr-value">{en(facts["strategy_bp"], 1, sign=True)} bp</div>
<p>per trade, after costs, for a strategy that follows the signal (2020-2026): forecasting
demand is not enough to make money.</p></div>
</div>
""",
    unsafe_allow_html=True,
)

st.subheader("More details")
links = [
    ("vues/prochaines.py", "Upcoming auctions", "the full table, for every measure"),
    ("vues/suivi.py", "Live track record", "frozen forecasts against actual results"),
    ("vues/backtest.py", "Model performance", "how well the models did since 2014"),
    ("vues/explorer.py", "History", "every maturity and every measure since 2009"),
    ("vues/qualite.py", "Data quality", "each source, how fresh it is, and its checks"),
    ("vues/methodologie.py", "Methodology", "sources, variables, validation, limits"),
]
left, right = st.columns(2)
for index, (path, label, text) in enumerate(links):
    column = left if index % 2 == 0 else right
    column.page_link(path, label=f"{label}: {text}", icon=":material/arrow_forward:")
