"""Page « Model performance » : backtest walk-forward, importance des variables et stratégie."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from tresor import app_data as ad
from tresor import app_ui as ui
from tresor.formats import en

st.title("Model performance")
st.caption(
    "Walk-forward test: every quarter since 2014, the models are trained on all past auctions"
    " and tested on the next quarter, never on data they have seen. The score is the"
    " out-of-sample R² against the average of the last 3 auctions: above 0, the model beats that"
    " simple rule."
)

metrics = ui.csv("backtest_metrics.csv")
backtest = ui.parquet("backtest_predictions.parquet")
family = st.segmented_control(
    "Securities", list(ad.FAMILY_LABELS), default="nominal_coupon",
    format_func=ad.FAMILY_LABELS.get, required=True,
) or "nominal_coupon"  # fmt: skip
tabs = st.tabs(["Scores", "Cumulative gain", "What drives the forecast", "Trading test"])


def interval(value: float, low: float, high: float) -> str:
    return f"{en(value * 100, 1, sign=True)}%  [{en(low * 100, 0)}, {en(high * 100, 0)}]"


with tabs[0]:
    period = st.segmented_control(
        "Period", ["2014-2026", "2014-2019", "2020-2026"], default="2020-2026", required=True,
        help="The best single model is picked on 2014-2019 and judged on 2020-2026.",
    ) or "2020-2026"  # fmt: skip
    data = metrics[
        (metrics["family"] == family)
        & (metrics["period"] == period)
        & (metrics["feature_set"] == "toutes")
    ]
    order = [name for name in ad.MODEL_LABELS if name in set(data["model"])]
    table = pd.DataFrame({"Model": [ad.MODEL_LABELS[name] for name in order]})
    for target, label in ad.TARGET_LABELS.items():
        part = data[data["target"] == target].set_index("model").reindex(order)
        table[label] = [
            interval(r, lo, hi)
            for r, lo, hi in zip(
                part["r2_vs_mean3"], part["r2_ci_low"], part["r2_ci_high"], strict=True
            )  # fmt: skip
        ]
    st.dataframe(table, hide_index=True, width="stretch", height=35 * (len(table) + 1) + 3)
    best = ui.csv("best_models.csv")
    chosen = best[best["family"] == family].set_index("target")["model"]
    st.caption(
        "In brackets: 90% confidence interval. Best single model picked on 2014-2019: "
        + ", ".join(f"{ad.MODEL_LABELS[m]} ({ad.SHORT_TARGETS[t]})" for t, m in chosen.items())
        + ". *Gap* models forecast the surprise relative to the last-3 average, as a robustness"
        " check. *Constant* is the average of the whole family, a very weak benchmark except for"
        " the tail."
    )
    with st.expander("Do the extra sources (dealers, CFTC, liquidity) help?"):
        comparison = ui.csv("new_sources_comparison.csv")
        rows = comparison[
            (comparison["family"] == family) & (comparison["model"] == "moyenne_5")
        ].set_index("target")
        st.dataframe(
            pd.DataFrame(
                {
                    "Measure": [ad.TARGET_LABELS[t] for t in rows.index],
                    "With": rows["toutes"].map(lambda v: en(v * 100, 1, sign=True, suffix="%")),
                    "Without": rows["sans_nouvelles_sources"].map(
                        lambda v: en(v * 100, 1, sign=True, suffix="%")
                    ),
                    "Difference (points)": [
                        f"{en(g * 100, 1, sign=True)}  [{en(lo * 100, 1)}, {en(hi * 100, 1)}]"
                        for g, lo, hi in zip(
                            rows["gain"], rows["gain_ci_low"], rows["gain_ci_high"], strict=True
                        )  # fmt: skip
                    ],
                }
            ),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "5-model average, 2014-2026. None of the differences is significantly positive: these"
            " sources do not improve the forecast, and slightly hurt some measures."
        )

with tabs[1]:
    target = st.selectbox("Measure", list(ad.TARGET_LABELS), format_func=ad.TARGET_LABELS.get)
    models = [m for m in ad.MODEL_LABELS if m in set(backtest["model"])]
    model = st.selectbox(
        "Model", models, format_func=ad.MODEL_LABELS.get,
        index=models.index("moyenne_5") if "moyenne_5" in models else 0,
    )  # fmt: skip
    gain = ad.cumulative_gain(backtest, family, target, model)
    figure = go.Figure()
    figure.add_scatter(x=gain["auction_date"], y=gain["gain"] * 100, mode="lines",
                       name=ad.MODEL_LABELS[model],
                       line={"color": ui.SERIES[0], "width": 2})  # fmt: skip
    if family == "bill":
        for label, day in ad.DEBT_CEILING.items():
            figure.add_vline(x=pd.Timestamp(day), line={"color": ui.MUTED, "dash": "dash"})
            figure.add_annotation(x=pd.Timestamp(day), y=1, yref="paper",
                                  text=f"debt ceiling {label}", showarrow=False,
                                  xanchor="left", font={"size": 11})  # fmt: skip
    figure.update_layout(title="Cumulative gain versus the last-3 average (ends at the R²)",
                         showlegend=False)  # fmt: skip
    figure.update_yaxes(ticksuffix="%")
    ui.chart(figure)
    note = (
        "Running total of how much smaller the model's squared errors are than those of the"
        " last-3 average, as a share of the latter. Above 0, the model does better."
    )
    if family == "bill":
        note += " For bills, part of the gain comes from the debt-ceiling episodes (dashed lines)."
    st.caption(note)

with tabs[2]:
    importance = ui.csv("feature_importance.csv")
    target = st.selectbox(
        "Measure", list(ad.TARGET_LABELS), format_func=ad.TARGET_LABELS.get, key="imp_target"
    )
    part = importance[(importance["family"] == family) & (importance["target"] == target)]
    if part.empty:
        st.info("No importance computed for this measure.")
    else:
        st.caption(
            f"Model picked on 2014-2019: {ad.MODEL_LABELS[part['model'].iloc[0]]}, trained before"
            " 2022 and explained on 2022-2026. *Permutation* shows what a variable really adds"
            " out of sample; *SHAP* shows how much it weighs in the model's formula. The two can"
            " differ."
        )
        left, right = st.columns(2)
        top = part.nlargest(12, "permutation_mean").iloc[::-1]
        figure = go.Figure(
            go.Bar(
                x=top["permutation_mean"],
                y=[ad.feature_label(name) for name in top["feature"]],
                orientation="h",
                marker_color=ui.SERIES[0],
                error_x={"type": "data", "array": top["permutation_std"], "color": ui.MUTED},
            )
        )
        figure.update_layout(title="Permutation: what really helps")
        figure.update_xaxes(title="Increase in squared error when the variable is shuffled")
        ui.chart(figure, height=480, container=left)
        top = part.nlargest(12, "shap_share").iloc[::-1]
        figure = go.Figure(
            go.Bar(
                x=top["shap_share"] * 100,
                y=[ad.feature_label(name) for name in top["feature"]],
                orientation="h",
                marker_color=ui.MUTED,
            )
        )
        figure.update_layout(title="SHAP: what weighs in the formula")
        figure.update_xaxes(title="Share of total contributions", ticksuffix="%")
        ui.chart(figure, height=480, container=right)

with tabs[3]:
    st.caption(
        "Notes and bonds only. The strategy is short duration (it gains if the yield rises) from"
        " the previous close to the auction-day close when the bid-to-cover forecast is below the"
        " last-3 average, and long otherwise; 0.5 bp of cost per trade. Results in basis points"
        " (bp) of yield."
    )
    quintiles = ui.csv("quintiles.csv")
    full = quintiles[quintiles["period"] == "2009-2026"]
    y_range = [full["ci_low"].min() - 0.4, full["ci_high"].max() + 0.4]
    columns = st.columns(2)
    for column, (name, label) in zip(columns, ad.SURPRISE_LABELS.items(), strict=True):
        rows = full[full["surprise"] == name].sort_values("quintile")
        figure = go.Figure(
            go.Bar(
                x=["1", "2", "3", "4", "5"],
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
        figure.update_layout(title=label.replace(" (", "<br>("))
        figure.update_xaxes(title="Surprise quintile (1 = strongest demand)")
        figure.update_yaxes(title="Yield change (bp)", range=y_range)
        ui.chart(figure, height=360, container=column)
    count = int(full.loc[full["surprise"] == "part_des_dealers", "n"].sum())
    st.caption(
        f"Notes and bonds, 2009-2026 ({en(count, 0)} auctions): average change in the yield of"
        " the auction's maturity from the previous close to the auction-day close, with a 90%"
        " interval. The weakest auctions come with a rise in yields that day."
    )
    summary = ui.csv("strategy_summary.csv")
    rows = summary[summary["period"] == "2014-2026"].set_index("rule")
    st.dataframe(
        pd.DataFrame(
            {
                "Rule": [ad.RULE_LABELS[rule] for rule in rows.index],
                "Trades": rows["trades"].map(lambda v: en(v, 0)),
                "Average P&L per trade (bp)": [
                    f"{en(m, 2, sign=True)}  [{en(lo, 2)}, {en(hi, 2)}]"
                    for m, lo, hi in zip(
                        rows["mean_bp"], rows["mean_ci_low"], rows["mean_ci_high"], strict=True
                    )  # fmt: skip
                ],
                "Before cost (bp)": rows["mean_gross_bp"].map(lambda v: en(v, 2, sign=True)),
                "Versus always short (bp)": [
                    "–" if pd.isna(g) else f"{en(g, 2, sign=True)}  [{en(lo, 2)}, {en(hi, 2)}]"
                    for g, lo, hi in zip(
                        rows["vs_vendeur_bp"],
                        rows["vs_vendeur_ci_low"],
                        rows["vs_vendeur_ci_high"],
                        strict=True,
                    )  # fmt: skip
                ],
                "Hit rate": rows["hit_rate"].map(lambda v: en(v * 100, 1, suffix="%")),
                "Total (bp)": rows["total_bp"].map(lambda v: en(v, 0, sign=True)),
                "Sharpe": rows["sharpe"].map(lambda v: en(v, 2, sign=True)),
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
                           name=ad.RULE_LABELS[rule], line=line)  # fmt: skip
    figure.update_layout(title="Cumulative P&L after costs (bp of yield)")
    ui.chart(figure, height=440)
    loss = -rows.loc["strategie", "mean_bp"]
    perfect = rows.loc["information_parfaite", "mean_bp"]
    st.caption(
        f"2014-2026. The strategy loses {en(loss, 1)} bp per trade after costs and does not beat"
        " always being short. Even knowing the auction result in advance would only earn"
        f" {en(perfect, 1, sign=True)} bp per trade after costs: the auction surprise explains"
        " only a small part of the day's yield move. Separately, the predictable part of demand"
        " has no measurable effect on yields."
    )
