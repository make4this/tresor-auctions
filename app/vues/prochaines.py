"""Page 1 : prochaines enchères, prévisions et signal."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from tresor import app_data as ad
from tresor import app_ui as ui
from tresor.config import SIGNAL_THRESHOLD

st.title("Prochaines enchères")
st.caption(
    "Prévisions de la moyenne des 5 modèles (modèle de référence fixé à l'avance). Une"
    " prévision est figée à partir du jour de l'enchère : elle n'est plus jamais modifiée."
)

log = ui.parquet("predictions_log.parquet")
auctions = ui.parquet("auctions.parquet", folder="clean")
table = ad.upcoming_table(log, auctions, ui.today_new_york())


def display(frame: pd.DataFrame) -> pd.DataFrame:
    """Table affichée, nombres à la française."""
    return pd.DataFrame(
        {
            "Date": [ui.date_fr(day) for day in frame["auction_date"]],
            "Enchère": frame["title"],
            "Montant": [
                ad.fr(amount / 1e9, 0, suffix=" Md$") + (" (estimé)" if estimated else "")
                for amount, estimated in zip(
                    frame["offering_amt"], frame["offering_estimated"], strict=True
                )
            ],
            "Bid-to-cover prévu": frame["pred_btc"].map(lambda v: ad.fr(v, 2)),
            "Moyenne des 3 dernières": frame["bench_mean3_btc"].map(lambda v: ad.fr(v, 2)),
            "Écart": frame["gap_btc"].map(lambda v: ad.fr(v, 3, sign=True)),
            "Signal": frame["signal"],
            "Part des dealers prévue": frame["pred_dealer_share"].map(
                lambda v: ad.fr(v * 100, 1, suffix=" %")
            ),
            "Tail prévu (pb)": frame["pred_tail_bp"].map(lambda v: ad.fr(v, 1, sign=True)),
        }
    )


def colour(value: str) -> str:
    if value == ad.WEAK:
        return f"color: {ui.ALERT}; font-weight: 600"
    if value == ad.STRONG:
        return f"color: {ui.PRIMARY}; font-weight: 600"
    return ""


if table.empty:
    st.info("Aucune enchère de notes, bonds ou bills n'est annoncée pour l'instant.")
else:
    shown = display(table)
    st.dataframe(
        shown.style.map(colour, subset=["Signal"]),
        hide_index=True,
        width="stretch",
        height=35 * (len(shown) + 1) + 3,  # toutes les enchères, sans défilement
    )
    threshold = ad.fr(SIGNAL_THRESHOLD, 2)
    st.caption(
        f"Signal : demande **plus faible** si le bid-to-cover prévu est inférieur d'au moins"
        f" {threshold} à la moyenne des 3 dernières enchères de même maturité, **plus forte**"
        f" s'il est supérieur d'au moins {threshold}, **dans la norme** sinon. L'écart est"
        " calculé sur les valeurs non arrondies. Montant « estimé » : pas encore annoncé par le"
        " Trésor, on reprend celui de la dernière enchère de même maturité."
    )

provisional = st.session_state.get("provisional")
if provisional is not None and not provisional.empty:
    with st.expander("Prévision provisoire de la dernière mise à jour (non inscrite au journal)"):
        fresh = ad.upcoming_table(
            provisional.assign(predicted_at=provisional["computed_at"]),
            auctions,
            ui.today_new_york(),
        )
        if not fresh.empty:
            styled = display(fresh).style.map(colour, subset=["Signal"])
            st.dataframe(styled, hide_index=True, width="stretch")

st.subheader("À lire avant d'utiliser ces prévisions")
record = ad.signal_track_record(ui.parquet("backtest_predictions.parquet")).set_index("family")
r2 = ad.backtest_r2(ui.csv("backtest_metrics.csv")).set_index(["family", "target"])
left, right = st.columns(2)
with left:
    st.markdown("**Fiabilité passée du signal** (backtest 2014-2026)")
    lines = []
    for family, label in ad.FAMILY_LABELS.items():
        if family in record.index:
            row = record.loc[family]
            lines.append(
                f"- {label} : « plus faible » sur {ad.fr(row['share_weak'] * 100, 0)} % des"
                f" enchères ; le bid-to-cover réel était ensuite sous la moyenne des 3 dans"
                f" {ad.fr(row['weak_then_below'] * 100, 0)} % des cas (contre"
                f" {ad.fr(row['base_rate_below'] * 100, 0)} % en moyenne)."
            )
    st.markdown("\n".join(lines))
with right:
    st.markdown("**Qualité des prévisions** (R² contre la moyenne des 3 dernières, 2020-2026)")
    lines = []
    for family, label in ad.FAMILY_LABELS.items():
        values = [
            f"{ad.TARGET_LABELS[target].split(' (')[0].lower()}"
            f" {ad.fr(r2.loc[(family, target), 'r2_vs_mean3'] * 100, 1, sign=True)} %"
            for target in ad.TARGET_LABELS
            if (family, target) in r2.index
        ]
        lines.append(f"- {label} : " + ", ".join(values) + ".")
    st.markdown("\n".join(lines))
st.warning(
    "**Part des dealers** : pour les notes et bonds, la moyenne des 5 modèles fait moins bien"
    " que la moyenne des 3 dernières enchères depuis 2020 (la part des dealers a fortement"
    " baissé et les modèles restent tirés vers le passé). **Tail** : écart entre le taux de"
    " l'enchère et le taux de clôture de la veille ; il reflète surtout le mouvement du marché"
    " le jour même et se prévoit à peine mieux qu'une constante."
)
