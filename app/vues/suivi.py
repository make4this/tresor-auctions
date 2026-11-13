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
    " 3 dernières enchères (le benchmark à battre)."
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
done = ad.live_results(tracking, target)

if done.empty:
    first = tracking["predicted_at"].min()
    start = ui.date_fr(first.tz_convert("America/New_York")) if pd.notna(first) else "–"
    st.info(
        f"Aucune enchère réalisée depuis le début du suivi ({start}). Les premiers résultats"
        " apparaîtront ici le soir de la première enchère suivie. En attendant, la page"
        " « Backtest » montre la même comparaison sur 2014-2026."
    )
else:
    left, right = st.columns(2)
    with left:
        figure = go.Figure()
        low = min(done["actual"].min(), done["pred"].min())
        high = max(done["actual"].max(), done["pred"].max())
        figure.add_scatter(x=[low, high], y=[low, high], mode="lines", name="Prévision parfaite",
                           line={"color": ui.MUTED, "dash": "dot", "width": 1})  # fmt: skip
        figure.add_scatter(
            x=done["pred"],
            y=done["actual"],
            mode="markers",
            name="Moyenne des 5",
            marker={"color": ui.SERIES[0], "size": 9},
            text=[ui.date_fr(day) for day in done["auction_date"]],
            hovertemplate="%{text}<br>prévu %{x:.3f}<br>réalisé %{y:.3f}<extra></extra>",
        )
        figure.add_scatter(
            x=done["bench_mean3"],
            y=done["actual"],
            mode="markers",
            name="Moyenne des 3 dernières",
            marker={"color": ui.SERIES[1], "size": 8, "symbol": "diamond"},
            hovertemplate="mean3 %{x:.3f}<br>réalisé %{y:.3f}<extra></extra>",
        )
        figure.update_layout(title="Prévu contre réalisé")
        figure.update_xaxes(title="Prévu")
        figure.update_yaxes(title="Réalisé")
        st.plotly_chart(ui.style(figure), width="stretch")
    with right:
        figure = go.Figure()
        line = {"color": ui.SERIES[0], "width": 2}
        figure.add_scatter(x=done["auction_date"], y=done["gain"], mode="lines+markers",
                           name="Gain cumulé", line=line)  # fmt: skip
        figure.update_layout(
            title="Gain cumulé contre la moyenne des 3 dernières (au-dessus de 0 : mieux)"
        )
        st.plotly_chart(ui.style(figure), width="stretch")
    st.dataframe(
        pd.DataFrame(
            {
                "Date": [ui.date_fr(day) for day in done["auction_date"]],
                "Enchère": [ad.MATURITY_LABELS.get(b, b) for b in done["maturity_bucket"]],
                "Prévu": done["pred"].map(lambda v: ad.fr(v, 3)),
                "Moyenne des 3": done["bench_mean3"].map(lambda v: ad.fr(v, 3)),
                "Réalisé": done["actual"].map(lambda v: ad.fr(v, 3)),
                "Erreur du modèle": done["error"].map(lambda v: ad.fr(v, 3, sign=True)),
                "Erreur de la moyenne des 3": done["bench_error"].map(
                    lambda v: ad.fr(v, 3, sign=True)
                ),
            }
        ),
        hide_index=True,
        width="stretch",
    )

pending = tracking[(tracking["model"] == "moyenne_5") & (tracking["status"] != "realisee")]
if not pending.empty:
    st.subheader("Prévisions en attente de résultat")
    wide = pending[pending["target"] == target].sort_values("auction_date")
    st.dataframe(
        pd.DataFrame(
            {
                "Date": [ui.date_fr(day) for day in wide["auction_date"]],
                "Enchère": [ad.MATURITY_LABELS.get(b, b) for b in wide["maturity_bucket"]],
                "Prévu": wide["pred"].map(lambda v: ad.fr(v, 3)),
                "Moyenne des 3": wide["bench_mean3"].map(lambda v: ad.fr(v, 3)),
                "Statut": wide["status"].map(
                    {"en_attente": "en attente", "sans_resultat": "sans résultat"}
                ),
                "Prévu le (New York)": [
                    ui.date_fr(t.tz_convert("America/New_York")) for t in wide["predicted_at"]
                ],
            }
        ),
        hide_index=True,
        width="stretch",
    )
