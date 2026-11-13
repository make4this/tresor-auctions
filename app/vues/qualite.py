"""Page 5 : qualité des données et rapport de nettoyage."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from tresor import app_data as ad
from tresor import app_ui as ui

st.title("Qualité des données")
quality = ui.json_file("data_quality.json")
report = ui.json_file("cleaning_report.json")
if not quality:
    st.error("Rapport de qualité absent : lancer `python -m tresor.pipeline --clean`.")
    st.stop()

STATUS = {"vert": "🟢 bon", "orange": "🟠 à surveiller", "rouge": "🔴 problème"}
st.markdown(
    f"**État général : {STATUS.get(str(quality.get('status')), '?')}** — contrôles du"
    f" {pd.Timestamp(str(quality.get('today_new_york'))):%d/%m/%Y} (date de New York)."
)

table = ad.quality_table(quality)
st.subheader("Sources")
st.dataframe(
    pd.DataFrame(
        {
            "Source": table["source"].map(lambda s: ad.SOURCE_LABELS.get(s, s)),
            "État": table["status"].map(lambda s: STATUS.get(s, s)),
            "Lignes": table["rows"].map(lambda v: ad.fr(v, 0)),
            "Dernière date": table["last_date"],
            "Jours ouvrés depuis": table["business_days_since_last"].map(lambda v: ad.fr(v, 0)),
            "Alertes": table["alerts"],
        }
    ),
    hide_index=True,
    width="stretch",
    height=35 * (len(table) + 1) + 3,  # toutes les sources, sans défilement
)
st.caption(
    "Une source est signalée si ses données ont plus de 5 jours ouvrés de retard ou si son"
    " nombre de lignes baisse anormalement. Le calendrier des enchères à venir a une date"
    " future, d'où un retard négatif. Fiscal Data publie les résultats avec plusieurs jours de"
    " retard : TreasuryDirect complète les enchères récentes."
)

auctions = quality.get("auctions", {})
market = quality.get("market", {})
st.subheader("Contrôles")
checks = [
    ("Enchères : clés en double", auctions.get("duplicate_keys")),
    ("Enchères passées sans résultat récent", len(auctions.get("recent_without_results", []))),
    ("Séries régulières interrompues", len(auctions.get("stale_regular_series", []))),
    ("Marché : clés en double", market.get("duplicate_keys")),
    ("Marché : taux hors bornes", len(market.get("rates_outside_bounds", []) or [])),
    (
        "Marché : valeurs sans date de publication",
        market.get("values_without_valid_publication_date"),
    ),
]
st.dataframe(
    pd.DataFrame({"Contrôle": [c for c, _ in checks], "Cas trouvés": [v for _, v in checks]}),
    hide_index=True,
    width="stretch",
)
for level, items in (
    ("Alertes", quality.get("alerts", [])),
    ("Avertissements", quality.get("warnings", [])),
):
    if items:
        st.warning(f"{level} : " + " ; ".join(map(str, items)))

if report:
    st.subheader("Rapport de nettoyage des enchères")
    cleaning = report.get("auctions", {})
    rows_in = cleaning.get("rows_in", {})
    st.markdown(
        "Lignes reçues : "
        + ", ".join(f"{name} {ad.fr(count, 0)}" for name, count in rows_in.items())
        + f". Doublons retirés : {ad.fr(cleaning.get('duplicates_removed', 0), 0)}."
        + f" Après fusion : {ad.fr(cleaning.get('rows_after_merge', 0), 0)} lignes."
    )
    excluded = cleaning.get("excluded", {})
    if excluded:
        st.dataframe(
            pd.DataFrame(
                {
                    "Exclusion": [ad.EXCLUSION_LABELS.get(name, name) for name in excluded],
                    "Lignes": [info.get("rows", 0) for info in excluded.values()],
                }
            ),
            hide_index=True,
            width="stretch",
        )
        with st.expander("Exemples de lignes exclues"):
            for name, info in excluded.items():
                st.markdown(f"**{ad.EXCLUSION_LABELS.get(name, name)}**")
                st.dataframe(pd.DataFrame(info.get("examples", [])), hide_index=True)
