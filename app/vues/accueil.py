"""Page d'accueil : le projet et les prochaines enchères, en mots simples.

Les détails (tableaux complets, backtest, qualité des données) sont dans les pages « Pour aller
plus loin ». Tous les chiffres sont lus dans les fichiers de data/, comme sur les autres pages.
"""

from __future__ import annotations

from html import escape

import streamlit as st

from tresor import app_data as ad
from tresor import app_ui as ui
from tresor.config import SIGNAL_THRESHOLD

st.markdown(
    """
<div class="tr-hero">
<h1>Les enchères du Trésor américain,<br><span class="tr-soft">prévues avant qu'elles aient
lieu</span></h1>
<p>Chaque semaine, les États-Unis empruntent des dizaines de milliards de dollars en vendant des
titres aux enchères. Ce site estime, avant chaque enchère, si les investisseurs seront plus ou
moins nombreux que d'habitude à vouloir acheter.</p>
</div>
""",
    unsafe_allow_html=True,
)

quality = ui.json_file("data_quality.json")
tables = len(quality.get("sources", {})) if isinstance(quality.get("sources"), dict) else 11
st.markdown(
    f"""
<div class="tr-steps">
<div class="tr-step"><span class="tr-number">1</span><h4>Collecter</h4>
<p>Chaque soir, {tables} tables de données publiques et gratuites (Trésor américain, Fed,
CFTC, CBOE) sont téléchargées automatiquement.</p></div>
<div class="tr-step"><span class="tr-number">2</span><h4>Prévoir</h4>
<p>Pour chaque enchère annoncée, 5 modèles estiment la demande à partir de ce qui était connu
la veille ; on retient leur moyenne.</p></div>
<div class="tr-step"><span class="tr-number">3</span><h4>Vérifier</h4>
<p>La prévision est figée le jour de l'enchère, puis comparée au vrai résultat, publié l'après-midi
même.</p></div>
</div>
""",
    unsafe_allow_html=True,
)

st.subheader("Les prochaines enchères")
log = ui.parquet("predictions_log.parquet")
auctions = ui.parquet("auctions.parquet", folder="clean")
cards = ad.home_cards(ad.upcoming_table(log, auctions, ui.today_new_york()))
FIRST_CARDS = 6  # les suivantes sont repliées : la page reste courte


def show_cards(selection: list[dict[str, str]]) -> None:
    blocks = [
        f"""<div class="tr-card"><span class="tr-date">{escape(ui.date_fr(card["date"]))} ·
{escape(card["family"])} · {escape(card["amount"])}</span><h4>{escape(card["title"])}</h4>
<span class="tr-badge {card["css"]}">{escape(card["badge"])}</span>
<p>{escape(card["sentence"])}</p></div>"""
        for card in selection
    ]
    st.markdown(f'<div class="tr-cards">{"".join(blocks)}</div>', unsafe_allow_html=True)


if not cards:
    st.info("Aucune enchère de notes, bonds ou bills n'est annoncée pour l'instant.")
else:
    show_cards(cards[:FIRST_CARDS])
    if len(cards) > FIRST_CARDS:
        with st.expander(f"Voir les {len(cards) - FIRST_CARDS} autres enchères annoncées"):
            show_cards(cards[FIRST_CARDS:])

threshold = ad.fr(SIGNAL_THRESHOLD, 2)
st.markdown(
    f"""
<div class="tr-panel">
<h4>Comment lire une carte</h4>
<p>Le <b>bid-to-cover</b> est le montant demandé par les investisseurs divisé par le montant
vendu : 2,8 veut dire qu'ils ont demandé 2,8 fois ce que le Trésor vendait. Plus il est haut,
plus la demande est forte.</p>
<p>La demande attendue est <b>plus faible</b> si le bid-to-cover prévu est au moins {threshold}
en dessous de la moyenne des 3 dernières enchères de même durée, <b>plus forte</b> s'il est au
moins {threshold} au-dessus, et <b>dans la norme</b> sinon.</p>
</div>
""",
    unsafe_allow_html=True,
)

st.subheader("Ce que montrent les tests depuis 2014")
facts = ad.key_facts(
    ui.csv("backtest_metrics.csv"), ui.csv("strategy_summary.csv"), ui.csv("quintiles.csv")
)
# R² contre mean3 = baisse de l'erreur quadratique : écrite comme une variation de l'erreur.
coupons = ad.fr(-facts["r2_coupons"] * 100, 0, sign=True, suffix=" %")
bills = ad.fr(-facts["r2_bills"] * 100, 0, sign=True, suffix=" %")
st.markdown(
    f"""
<div class="tr-facts">
<div class="tr-fact"><div class="tr-value">{coupons}</div>
<p>d'erreur (au carré) par rapport à la règle simple « comme les 3 dernières fois », pour les
notes et bonds ; {bills} pour les bills (2020-2026). La prévision aide, un peu.</p></div>
<div class="tr-fact"><div class="tr-value">{ad.fr(facts["weak_move_bp"], 1, sign=True)} pb</div>
<p>de hausse moyenne du taux le jour de l'enchère (1 pb = 0,01 point de pourcentage), pour les
20 % d'enchères de notes et bonds où la demande est la plus faible par rapport à d'habitude
(2009-2026).</p></div>
<div class="tr-fact"><div class="tr-value">{ad.fr(facts["strategy_bp"], 1, sign=True)} pb</div>
<p>par opération, après coûts, pour une stratégie qui suit le signal (2020-2026) : prévoir la
demande ne suffit pas à gagner de l'argent.</p></div>
</div>
""",
    unsafe_allow_html=True,
)

st.subheader("Pour aller plus loin")
links = [
    ("vues/prochaines.py", "Prochaines enchères", "le tableau complet, avec toutes les cibles"),
    ("vues/suivi.py", "Suivi en direct", "les prévisions figées face aux vrais résultats"),
    ("vues/backtest.py", "Backtest", "la fiabilité des modèles, testée depuis 2014"),
    ("vues/explorer.py", "Explorer", "l'historique de chaque durée et de chaque cible"),
    ("vues/qualite.py", "Qualité des données", "chaque source, sa fraîcheur et ses contrôles"),
    ("vues/methodologie.py", "Méthodologie", "sources, variables, validation, limites"),
]
left, right = st.columns(2)
for index, (path, label, text) in enumerate(links):
    column = left if index % 2 == 0 else right
    column.page_link(path, label=f"{label} : {text}", icon=":material/arrow_forward:")

st.caption(ui.DISCLAIMER)
