"""Page 3 : backtest walk-forward, importance des variables et stratégie."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from tresor import app_data as ad
from tresor import app_ui as ui
from tresor import plots
from tresor.strategy import RULES

st.title("Backtest")
st.caption(
    "Validation walk-forward : chaque trimestre depuis 2014, les modèles sont entraînés sur tout"
    " le passé puis testés sur le trimestre suivant. R² hors échantillon contre la moyenne des 3"
    " dernières enchères (`mean3`) : au-dessus de 0, le modèle fait mieux que ce benchmark."
)

metrics = ui.csv("backtest_metrics.csv")
backtest = ui.parquet("backtest_predictions.parquet")
family = st.segmented_control(
    "Famille", list(ad.FAMILY_LABELS), default="nominal_coupon",
    format_func=ad.FAMILY_LABELS.get, required=True,
)  # fmt: skip
tabs = st.tabs(["Métriques", "Gain cumulé", "Variables", "Stratégie"])

with tabs[0]:
    period = st.segmented_control(
        "Période", ["2014-2026", "2014-2019", "2020-2026"], default="2020-2026", required=True,
        help="Le modèle « choisi » l'est sur 2014-2019 et jugé sur 2020-2026.",
    )  # fmt: skip
    data = metrics[
        (metrics["family"] == family)
        & (metrics["period"] == period)
        & (metrics["feature_set"] == "toutes")
    ]
    order = [name for name in plots.MODEL_LABELS if name in set(data["model"])]
    table = pd.DataFrame({"Modèle": [plots.MODEL_LABELS[name] for name in order]})
    for target, label in ad.TARGET_LABELS.items():
        part = data[data["target"] == target].set_index("model").reindex(order)
        table[label.split(" (")[0]] = [
            f"{ad.fr(r * 100, 1, sign=True)} %  [{ad.fr(lo * 100, 0)} ; {ad.fr(hi * 100, 0)}]"
            for r, lo, hi in zip(
                part["r2_vs_mean3"], part["r2_ci_low"], part["r2_ci_high"], strict=True
            )  # fmt: skip
        ]
    st.dataframe(table, hide_index=True, width="stretch", height=35 * (len(table) + 1) + 3)
    best = ui.csv("best_models.csv")
    chosen = best[best["family"] == family].set_index("target")["model"]
    st.caption(
        "R² contre `mean3`, avec son intervalle à 90 % (bootstrap par année). Modèle choisi sur"
        " 2014-2019 : "
        + ", ".join(
            f"{ad.TARGET_LABELS[t].split(' (')[0].lower()} : {plots.MODEL_LABELS[m]}"
            for t, m in chosen.items()
        )
        + ". La « Constante » est la moyenne de toute la famille, toutes maturités confondues :"
        " un benchmark très faible, sauf pour le tail."
    )

with tabs[1]:
    target = st.selectbox("Cible", list(ad.TARGET_LABELS), format_func=ad.TARGET_LABELS.get)
    models = [m for m in plots.MODEL_LABELS if m in set(backtest["model"])]
    model = st.selectbox("Modèle", models, index=models.index("moyenne_5"),
                         format_func=plots.MODEL_LABELS.get)  # fmt: skip
    gain = ad.cumulative_gain(backtest, family, target, model)
    figure = go.Figure()
    figure.add_scatter(x=gain["auction_date"], y=gain["gain"] * 100, mode="lines",
                       name=plots.MODEL_LABELS[model],
                       line={"color": ui.SERIES[0], "width": 2})  # fmt: skip
    figure.update_layout(
        title="Gain cumulé contre mean3, en % de l'erreur totale de mean3 (finit au R²)",
        showlegend=False,
    )
    figure.update_yaxes(ticksuffix=" %")
    st.plotly_chart(ui.style(figure), width="stretch")
    st.caption(
        "Somme cumulée de (réel − mean3)² − (réel − prévu)², divisée par la somme des"
        " (réel − mean3)². Pour les bills, les sauts d'octobre 2015 et d'octobre 2021 viennent des"
        " crises du plafond de la dette."
    )

with tabs[2]:
    importance = ui.csv("feature_importance.csv")
    target = st.selectbox(
        "Cible", list(ad.TARGET_LABELS), format_func=ad.TARGET_LABELS.get, key="imp_target"
    )
    part = importance[(importance["family"] == family) & (importance["target"] == target)]
    if part.empty:
        st.info("Pas d'importance calculée pour cette cible.")
    else:
        st.caption(
            f"Modèle choisi sur 2014-2019 : {plots.MODEL_LABELS[part['model'].iloc[0]]},"
            " entraîné avant 2022 et expliqué sur 2022-2026."
        )
        left, right = st.columns(2)
        top = part.nlargest(12, "permutation_mean").iloc[::-1]
        figure = go.Figure(
            go.Bar(
                x=top["permutation_mean"],
                y=[plots.feature_label(name) for name in top["feature"]],
                orientation="h",
                marker_color=ui.SERIES[0],
                error_x={"type": "data", "array": top["permutation_std"], "color": ui.MUTED},
            )
        )
        figure.update_layout(title="Permutation : ce qui aide vraiment (hors échantillon)")
        left.plotly_chart(ui.style(figure, height=460), width="stretch")
        top = part.nlargest(12, "shap_share").iloc[::-1]
        figure = go.Figure(
            go.Bar(
                x=top["shap_share"] * 100,
                y=[plots.feature_label(name) for name in top["feature"]],
                orientation="h",
                marker_color=ui.MUTED,
            )
        )
        figure.update_layout(title="SHAP : ce qui pèse dans la formule (part du total)")
        figure.update_xaxes(ticksuffix=" %")
        right.plotly_chart(ui.style(figure, height=460), width="stretch")

with tabs[3]:
    st.caption(
        "Notes et bonds. Vendeur de duration de la clôture de la veille à celle du jour si le"
        " modèle prévoit un bid-to-cover sous `mean3`, acheteur sinon ; 0,5 pb de coût par trade."
    )
    quintiles = ui.csv("quintiles.csv")
    full = quintiles[quintiles["period"] == "2009-2026"]
    columns = st.columns(2)
    # Même échelle pour les deux graphiques, qu'on compare côte à côte.
    y_range = [full["ci_low"].min() - 0.4, full["ci_high"].max() + 0.4]
    for column, (name, label) in zip(columns, plots.SURPRISE_LABELS.items(), strict=True):
        rows = full[full["surprise"] == name].sort_values("quintile")
        figure = go.Figure(
            go.Bar(
                x=["1 (forte)", "2", "3", "4", "5 (faible)"],
                y=rows["rate_change_mean_bp"],
                marker_color=ui.SERIES[0],
                error_y={
                    "type": "data",
                    "symmetric": False,
                    "array": rows["ci_high"] - rows["rate_change_mean_bp"],
                    "arrayminus": rows["rate_change_mean_bp"] - rows["ci_low"],
                    "color": ui.MUTED,
                },
            )
        )
        figure.update_layout(title=label)
        figure.update_yaxes(title="Variation du taux (pb)", range=y_range)
        column.plotly_chart(ui.style(figure, height=330), width="stretch")
    st.caption("Les enchères les plus faibles s'accompagnent d'une hausse du taux le jour même.")
    summary = ui.csv("strategy_summary.csv")
    rows = summary[summary["period"] == "2014-2026"].set_index("rule")
    st.dataframe(
        pd.DataFrame(
            {
                "Règle": [RULES[rule] for rule in rows.index],
                "Trades": rows["trades"].map(lambda v: ad.fr(v, 0)),
                "P&L moyen (pb)": [
                    f"{ad.fr(m, 2, sign=True)}  [{ad.fr(lo, 2)} ; {ad.fr(hi, 2)}]"
                    for m, lo, hi in zip(
                        rows["mean_bp"], rows["mean_ci_low"], rows["mean_ci_high"], strict=True
                    )  # fmt: skip
                ],
                "Avant coût (pb)": rows["mean_gross_bp"].map(lambda v: ad.fr(v, 2, sign=True)),
                "Réussite": rows["hit_rate"].map(lambda v: ad.fr(v * 100, 1, suffix=" %")),
                "Total (pb)": rows["total_bp"].map(lambda v: ad.fr(v, 0, sign=True)),
                "Sharpe": rows["sharpe"].map(lambda v: ad.fr(v, 2, sign=True)),
            }
        ),
        hide_index=True,
        width="stretch",
    )
    trades = ui.parquet("strategy_trades.parquet").sort_values("auction_date")
    figure = go.Figure()
    styles = {"strategie": (ui.SERIES[0], "solid"), "toujours_vendeur": (ui.SERIES[1], "solid"),
              "information_parfaite": (ui.MUTED, "dash")}  # fmt: skip
    for rule, (color, dash) in styles.items():
        line = {"color": color, "dash": dash, "width": 2}
        figure.add_scatter(x=trades["auction_date"], y=trades[rule].cumsum(), mode="lines",
                           name=RULES[rule], line=line)  # fmt: skip
    figure.update_layout(title="P&L cumulé après coûts (pb de taux)")
    st.plotly_chart(ui.style(figure, height=420), width="stretch")
    st.caption(
        "La stratégie perd de l'argent et ne fait pas mieux que « toujours vendeur » (écart non"
        " significatif). Même en connaissant le résultat de l'enchère à l'avance, le gain reste"
        " faible : la partie prévisible de la demande n'a pas d'effet mesurable sur le taux."
    )
