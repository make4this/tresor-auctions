"""Page 1 : prochaines enchères, prévisions et signal."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from tresor import app_data as ad
from tresor import app_ui as ui
from tresor.config import SIGNAL_THRESHOLD

st.title("Prochaines enchères")
st.caption(
    "Le **bid-to-cover** est le montant demandé divisé par le montant émis : plus il est élevé,"
    " plus la demande est forte. Les prévisions sont celles de la moyenne des 5 modèles,"
    " retenue pour ne pas choisir un modèle d'après ses résultats. Une prévision est figée à"
    " partir du jour de l'enchère : elle n'est plus jamais modifiée."
)

log = ui.parquet("predictions_log.parquet")
auctions = ui.parquet("auctions.parquet", folder="clean")
today = ui.today_new_york()
table = ad.upcoming_table(log, auctions, today)


def display(frame: pd.DataFrame) -> pd.DataFrame:
    """Table affichée : l'essentiel d'abord (lisible sur téléphone), nombres à la française."""
    return pd.DataFrame(
        {
            "Date": [ui.date_fr(day, year=False) for day in frame["auction_date"]],
            "Enchère": frame["title"],
            "Demande attendue": frame["signal"],
            "Écart": frame["gap_btc"].map(lambda v: ad.fr(v, 3, sign=True)),
            "Bid-to-cover prévu": frame["pred_btc"].map(lambda v: ad.fr(v, 2)),
            "Bid-to-cover, moyenne des 3 dernières": frame["bench_mean3_btc"].map(
                lambda v: ad.fr(v, 2)
            ),
            "Montant": [
                ad.fr(amount / 1e9, 0, suffix=" Md$") + (" (estimé)" if estimated else "")
                for amount, estimated in zip(
                    frame["offering_amt"], frame["offering_estimated"], strict=True
                )
            ],
            "Part des dealers prévue": frame["pred_dealer_share"].map(
                lambda v: ad.format_target(v, "dealer_share")
            ),
            "Tail prévu": frame["pred_tail_bp"].map(lambda v: ad.format_target(v, "tail_bp")),
        }
    )


def colour(value: str) -> str:
    if value == ad.WEAK:
        return f"color: {ui.ALERT}; font-weight: 600"
    if value == ad.STRONG:
        return f"color: {ui.PRIMARY}; font-weight: 600"
    return ""


def show(frame: pd.DataFrame) -> None:
    shown = display(frame)
    st.dataframe(
        shown.style.map(colour, subset=["Demande attendue"]),
        hide_index=True,
        width="stretch",
        height=35 * (len(shown) + 1) + 3,  # toutes les enchères, sans défilement
    )


if table.empty:
    st.info("Aucune enchère de notes, bonds ou bills n'est annoncée pour l'instant.")
else:
    show(table)
    threshold = ad.fr(SIGNAL_THRESHOLD, 2)
    st.caption(
        f"Demande attendue **plus faible** si le bid-to-cover prévu est inférieur d'au moins"
        f" {threshold} à la moyenne des 3 dernières enchères de même maturité, **plus forte** s'il"
        f" est supérieur d'au moins {threshold}, **dans la norme** sinon (écart calculé sur les"
        " valeurs non arrondies). Montant « estimé » : pas encore annoncé par le Trésor, on"
        " reprend celui de la dernière enchère de même maturité. Les prévisions des enchères"
        " lointaines sont recalculées chaque soir jusqu'à la veille (montant annoncé, résultats"
        " des enchères précédentes) : seule la dernière, figée le jour de l'enchère, est"
        " comparable aux résultats du backtest ci-dessous."
    )

provisional = st.session_state.get("provisional")
if provisional is not None and not provisional.empty:
    computed = pd.Timestamp(provisional["computed_at"].iloc[0]).tz_convert("America/New_York")
    title = (
        f"Prévision provisoire du {ui.date_fr(computed)} à {computed:%H:%M} (New York),"
        " non inscrite au journal"
    )
    with st.expander(title):
        fresh = ad.upcoming_table(
            provisional.assign(predicted_at=provisional["computed_at"]), auctions, today
        )
        if fresh.empty:
            st.write("Aucune enchère à venir dans cette mise à jour.")
        else:
            show(fresh)

st.subheader("À lire avant d'utiliser ces prévisions")
st.warning(
    f"{ui.DISCLAIMER} Le signal décrit la demande attendue à l'enchère, pas l'évolution des"
    " taux : une stratégie fondée sur lui a perdu de l'argent sur 2014-2026 (page Backtest)."
)
record = ad.signal_track_record(ui.parquet("backtest_predictions.parquet")).set_index("family")
r2 = ad.backtest_r2(ui.csv("backtest_metrics.csv")).set_index(["family", "target"])
left, right = st.columns(2)
with left:
    st.markdown("**Fiabilité passée du signal** (backtest 2014-2026)")
    lines = []
    for family, label in ad.FAMILY_LABELS.items():
        if family not in record.index:
            continue
        row = record.loc[family]
        lines.append(
            f"- {label} : « plus faible » sur {ad.fr(row['share_weak'] * 100, 0)} % des enchères"
            f" ({ad.fr(row['weak_count'], 0)}) ; dans {ad.fr(row['weak_then_below'] * 100, 0)} %"
            " de ces cas, le bid-to-cover réel a bien été inférieur à la moyenne des 3 dernières"
            f" (contre {ad.fr(row['base_rate_below'] * 100, 0)} % pour l'ensemble des enchères)."
            f" « Plus forte » sur {ad.fr(row['share_strong'] * 100, 0)} %"
            f" ({ad.fr(row['strong_count'], 0)}) ; supérieur ensuite dans"
            f" {ad.fr(row['strong_then_above'] * 100, 0)} % des cas"
            f" (contre {ad.fr(row['base_rate_above'] * 100, 0)} %)."
        )
    st.markdown("\n".join(lines))
with right:
    st.markdown(
        "**Qualité des prévisions** (R² contre la moyenne des 3 dernières, 2020-2026,"
        " intervalle à 90 %)"
    )

    def value(family: str, target: str) -> str:
        row = r2.loc[(family, target)]
        return (
            f"{ad.fr(row['r2_vs_mean3'] * 100, 1, sign=True)} % [{ad.fr(row['r2_ci_low'] * 100, 0)}"
            f" ; {ad.fr(row['r2_ci_high'] * 100, 0)}]"
        )

    lines = []
    for family, label in ad.FAMILY_LABELS.items():
        parts = [
            f"{ad.SHORT_TARGETS[target]} {value(family, target)}"
            for target in ("btc", "dealer_share", "indirect_share")
            if (family, target) in r2.index
        ]
        lines.append(f"- {label} : " + ", ".join(parts) + ".")
    st.markdown("\n".join(lines))
    st.caption(
        "Au-dessus de 0, la prévision fait mieux que la moyenne des 3 dernières enchères. Seul le"
        " bid-to-cover des notes et bonds est un résultat solide ; pour les bills, le gain est en"
        " partie gonflé par les crises du plafond de la dette (2015, 2021, 2023). Un intervalle"
        " qui contient 0 signifie que la différence n'est pas établie."
    )
tail_vs_constant = r2.loc[("nominal_coupon", "tail_bp"), "r2_vs_constant"]
st.warning(
    "**Part des dealers** : pour les notes et bonds, la moyenne des 5 modèles fait moins bien que"
    " la moyenne des 3 dernières enchères depuis 2020 (la part des dealers a fortement baissé et"
    " les modèles restent tirés vers le passé). **Tail** (écart entre le taux de l'enchère et le"
    " taux de clôture de la veille) : il reflète surtout le mouvement du marché le jour même. Il"
    " ne se prévoit qu'à peine mieux qu'une constante pour les notes et bonds"
    f" ({ad.fr(tail_vs_constant * 100, 1, sign=True)} %) ; pour les bills, le gain tient presque"
    " entièrement à la crise du plafond de la dette de 2023."
)
