"""Page 2 : suivi en direct, prévu contre réalisé."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from tresor import app_data as ad
from tresor import app_ui as ui

st.title("Suivi en direct")
st.caption(
    "Chaque prévision figée est comparée au résultat publié par le Trésor, et à la moyenne des"
    " 3 dernières enchères de même maturité (le benchmark à battre). Prévisions de la moyenne"
    " des 5 modèles."
)

tracking = ui.parquet("live_tracking.parquet")
counts = ad.live_summary(tracking)
columns = st.columns(4)
columns[0].metric("Enchères suivies", counts["suivies"])
columns[1].metric("Réalisées", counts["realisees"])
columns[2].metric("En attente", counts["en_attente"])
columns[3].metric("Sans résultat", counts["sans_resultat"], help="Date passée sans résultat :"
                  " enchère déplacée, annulée, ou résultat pas encore collecté.")  # fmt: skip

target = st.segmented_control(
    "Cible",
    list(ad.TARGET_LABELS),
    default="btc",
    format_func=lambda name: ad.TARGET_LABELS[name].split(" (")[0],
    required=True,
)
target = target or "btc"
scale = 100 if target in ("dealer_share", "indirect_share") else 1
done = ad.live_results(tracking, target)


def shown(values: pd.Series, sign: bool = False) -> pd.Series:
    return values.map(lambda v: ad.format_target(v, target, sign=sign))


if done.empty:
    pending = tracking[(tracking["model"] == "moyenne_5") & (tracking["status"] == "en_attente")]
    if counts["realisees"]:
        st.info("Des enchères ont eu lieu, mais aucun résultat n'est publié pour cette cible.")
    else:
        first = pending["auction_date"].min()
        when = f" ({ui.date_fr(first)})" if pd.notna(first) else ""
        st.info(
            f"Aucune enchère suivie n'a encore eu lieu. Les premiers résultats apparaîtront ici"
            f" après la première enchère suivie{when}, à la mise à jour du soir. La page"
            " « Backtest » montre la même comparaison sur 2014-2026."
        )
else:
    left, right = st.columns(2)
    figure = go.Figure()
    low = min(done["actual"].min(), done["pred"].min()) * scale
    high = max(done["actual"].max(), done["pred"].max()) * scale
    figure.add_scatter(x=[low, high], y=[low, high], mode="lines", name="Prévision parfaite",
                       line={"color": ui.MUTED, "dash": "dot", "width": 1})  # fmt: skip
    figure.add_scatter(
        x=done["pred"] * scale,
        y=done["actual"] * scale,
        mode="markers",
        name="Moyenne des 5",
        marker={"color": ui.SERIES[0], "size": 9},
        text=[ui.date_fr(day) for day in done["auction_date"]],
        hovertemplate="%{text}<br>prévu %{x:.2f}<br>réalisé %{y:.2f}<extra></extra>",
    )
    figure.add_scatter(
        x=done["bench_mean3"] * scale,
        y=done["actual"] * scale,
        mode="markers",
        name="Moyenne des 3 dernières",
        marker={"color": ui.SERIES[1], "size": 8, "symbol": "diamond"},
        hovertemplate="moyenne des 3 %{x:.2f}<br>réalisé %{y:.2f}<extra></extra>",
    )
    unit = " (%)" if scale == 100 else (" (pb)" if target == "tail_bp" else "")
    figure.update_layout(title="Prévu contre réalisé")
    figure.update_xaxes(title=f"Prévu{unit}")
    figure.update_yaxes(title=f"Réalisé{unit}")
    ui.chart(figure, container=left)
    figure = go.Figure()
    line = {"color": ui.SERIES[0], "width": 2}
    figure.add_scatter(x=done["auction_date"], y=done["gain"] * 100, mode="lines+markers",
                       name="Gain cumulé", line=line)  # fmt: skip
    figure.update_layout(
        title="Gain cumulé contre la moyenne des 3 dernières, en % de son erreur totale",
        showlegend=False,
    )
    figure.update_yaxes(ticksuffix=" %")
    ui.chart(figure, container=right)
    st.caption(
        "Gain cumulé : somme des (réel − moyenne des 3)² − (réel − prévu)², rapportée à l'erreur"
        " totale de la moyenne des 3. Au-dessus de 0, la prévision fait mieux ; la courbe finit"
        " au R² du suivi en direct."
    )
    st.dataframe(
        pd.DataFrame(
            {
                "Date": [ui.date_fr(day) for day in done["auction_date"]],
                "Enchère": [ad.MATURITY_LABELS.get(b, b) for b in done["maturity_bucket"]],
                "Prévu": shown(done["pred"]),
                "Moyenne des 3 dernières": shown(done["bench_mean3"]),
                "Réalisé": shown(done["actual"]),
                "Erreur de la prévision": shown(done["error"], sign=True),
                "Erreur de la moyenne des 3": shown(done["bench_error"], sign=True),
            }
        ),
        hide_index=True,
        width="stretch",
    )

waiting = tracking[(tracking["model"] == "moyenne_5") & (tracking["status"] != "realisee")]
waiting = waiting[waiting["target"] == target].sort_values("auction_date")
if not waiting.empty:
    st.subheader("Prévisions en attente de résultat")
    st.dataframe(
        pd.DataFrame(
            {
                "Date": [ui.date_fr(day) for day in waiting["auction_date"]],
                "Enchère": [ad.MATURITY_LABELS.get(b, b) for b in waiting["maturity_bucket"]],
                "Prévu": shown(waiting["pred"]),
                "Moyenne des 3 dernières": shown(waiting["bench_mean3"]),
                "Statut": waiting["status"].map(
                    {"en_attente": "en attente", "sans_resultat": "sans résultat"}
                ),
                "Prévu le (New York)": [
                    ui.date_fr(t.tz_convert("America/New_York")) for t in waiting["predicted_at"]
                ],
            }
        ),
        hide_index=True,
        width="stretch",
    )
