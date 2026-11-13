"""Page 5 : qualité des données et rapport de nettoyage."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from tresor import app_data as ad
from tresor import app_ui as ui
from tresor.clean.checks import MAX_DELAY_BUSINESS_DAYS, MAX_DELAY_OVERRIDES

st.title("Qualité des données")
quality = ui.json_file("data_quality.json")
report = ui.json_file("cleaning_report.json")
if not quality:
    st.error("Rapport de qualité absent : lancer `python -m tresor.pipeline --clean`.")
    st.stop()

STATUS = {"vert": "🟢 bon", "orange": "🟠 à surveiller", "rouge": "🔴 problème"}


def count(value: object) -> int:
    """Nombre de cas d'un contrôle, qu'il soit écrit comme un nombre ou comme une liste."""
    if isinstance(value, list | dict):
        return len(value)
    try:
        return int(value)  # type: ignore[call-overload]
    except (TypeError, ValueError):
        return 0


st.markdown(
    f"**État général : {STATUS.get(str(quality.get('status')), '?')}** — contrôles du"
    f" {ui.day_fr(quality.get('today_new_york'))} (date de New York)."
)

table = ad.quality_table(quality)
st.subheader("Sources")
if not table.empty:
    st.dataframe(
        pd.DataFrame(
            {
                "Source": table["source"].map(lambda s: ad.SOURCE_LABELS.get(s, s)),
                "État": table["status"].map(lambda s: STATUS.get(str(s), str(s))),
                "Lignes": table["rows"].map(lambda v: ad.fr(v, 0)),
                "Dernière observation": table["last_date"].map(ui.day_fr),
                "Dernière publication": table["last_published"].map(ui.day_fr),
                "Jours ouvrés depuis la publication": table["business_days_since_last"].map(
                    lambda v: ad.fr(v, 0)
                ),
                "Alertes": table["alerts"],
            }
        ),
        hide_index=True,
        width="stretch",
        height=35 * (len(table) + 1) + 3,  # toutes les sources, sans défilement
    )
overrides = ", ".join(
    f"{ad.fr(days, 0)} pour {ad.SOURCE_LABELS.get(name, name)}"
    for name, days in MAX_DELAY_OVERRIDES.items()
)
st.caption(
    f"Une source est signalée si sa dernière publication date de plus de"
    f" {MAX_DELAY_BUSINESS_DAYS} jours ouvrés ({overrides}, que TreasuryDirect complète), ou si"
    " son nombre de lignes baisse anormalement. Le calendrier des enchères à venir a des dates"
    " futures, d'où un retard négatif. Les séries hebdomadaires (dealers, CFTC) sont publiées"
    " plusieurs jours après leur date d'observation."
)

auctions = quality.get("auctions", {})
market = quality.get("market", {})
auctions = auctions if isinstance(auctions, dict) else {}
market = market if isinstance(market, dict) else {}
st.subheader("Contrôles")
checks = [
    ("Enchères : clés en double", auctions.get("duplicate_keys")),
    ("Enchères passées sans résultat récent", auctions.get("recent_without_results")),
    ("Séries régulières interrompues", auctions.get("stale_regular_series")),
    ("Marché : clés en double", market.get("duplicate_keys")),
    ("Marché : taux hors bornes", market.get("rates_outside_bounds")),
    (
        "Marché : valeurs sans date de publication",
        market.get("values_without_valid_publication_date"),
    ),
]
st.dataframe(
    pd.DataFrame(
        {"Contrôle": [c for c, _ in checks], "Cas trouvés": [count(v) for _, v in checks]}
    ),
    hide_index=True,
    width="stretch",
)
for level, items in (
    ("Alertes", quality.get("alerts", [])),
    ("Avertissements", quality.get("warnings", [])),
):
    if items:
        st.warning(f"{level} : " + " ; ".join(map(str, items)))

cleaning = report.get("auctions", {}) if isinstance(report, dict) else {}
if cleaning:
    st.subheader("Rapport de nettoyage des enchères")
    rows_in = cleaning.get("rows_in", {})
    excluded = cleaning.get("excluded", {})
    merged = int(cleaning.get("rows_after_merge", 0))
    dropped = sum(int(info.get("rows", 0)) for info in excluded.values())
    st.markdown(
        "Lignes reçues : "
        + ", ".join(f"{ad.INPUT_LABELS.get(n, n)} {ad.fr(c, 0)}" for n, c in rows_in.items())
        + f". Enchères présentes dans plusieurs sources (fusionnées) :"
        f" {ad.fr(cleaning.get('duplicates_removed', 0), 0)}. Après fusion : {ad.fr(merged, 0)}"
        f" ; après exclusions : **{ad.fr(merged - dropped, 0)} enchères**."
    )
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
                examples = pd.DataFrame(info.get("examples", []))
                if examples.empty:
                    continue
                shown = pd.DataFrame(
                    {
                        "CUSIP": examples.get("cusip"),
                        "Date": examples.get("auction_date", pd.Series(dtype=str)).map(ui.day_fr),
                        "Type": examples.get("security_type"),
                        "Durée": examples.get("security_term"),
                        "Montant": examples.get("offering_amt", pd.Series(dtype=float)).map(
                            lambda v: ad.fr(v / 1e6, 0, suffix=" M$")
                        ),
                    }
                )
                st.dataframe(shown, hide_index=True, width="stretch")
