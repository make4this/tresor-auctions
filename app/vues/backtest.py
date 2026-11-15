"""Page « Model performance » : backtest walk-forward, importance des variables et stratégie."""

from __future__ import annotations

from html import escape

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from tresor import app_data as ad
from tresor import app_ui as ui
from tresor.formats import en

ui.page_header(
    "Model performance",
    "Walk-forward test: every quarter since 2014, the models are trained on all past auctions"
    " and tested on the next quarter, never on data they have seen. The score is the"
    " out-of-sample R² against the average of the last 3 auctions: above 0, the model beats that"
    " simple rule.",
)

metrics = ui.csv("backtest_metrics.csv")
backtest = ui.parquet("backtest_predictions.parquet")
family = st.segmented_control(
    "Securities", list(ad.FAMILY_LABELS), default="nominal_coupon",
    format_func=ad.FAMILY_LABELS.get, required=True,
) or "nominal_coupon"  # fmt: skip
tabs = st.tabs(["Scores", "Cumulative gain", "What drives the forecast", "Trading test"])
VERDICT_GUIDE = (
    "<b>Green</b>: better than the last-3 average with 90% confidence. <b>Red</b>: worse."
    " <b>Grey</b>: no clear difference."
)


def interval(value: float, low: float, high: float) -> str:
    return f"{en(value * 100, 1, sign=True)}% [{en(low * 100, 0)}, {en(high * 100, 0)}]"


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
    rows = []
    for model in order:
        part = data[data["model"] == model].set_index("target")
        cells: list[str | tuple[str, str]] = [f"<b>{escape(ad.MODEL_LABELS[model])}</b>"]
        for target in ad.TARGET_LABELS:
            if target not in part.index:
                cells.append(("–", "tr-unclear"))
                continue
            row = part.loc[target]
            text = interval(row["r2_vs_mean3"], row["r2_ci_low"], row["r2_ci_high"])
            cells.append((text, f"tr-{ad.verdict(row['r2_ci_low'], row['r2_ci_high'])}"))
        rows.append(cells)
    best = ui.csv("best_models.csv")
    chosen = best[best["family"] == family].set_index("target")["model"]
    picked = ", ".join(f"{ad.MODEL_LABELS[m]} ({ad.SHORT_TARGETS[t]})" for t, m in chosen.items())
    st.markdown(
        ui.panel(
            "Scores by model",
            ui.table(["Model", *ad.TARGET_LABELS.values()], rows, right={1, 2, 3, 4}),
            f"Best single model picked on 2014-2019: {escape(picked)}.",
            [
                "Each cell: out-of-sample R² against the last-3 average, with its 90% confidence"
                " interval in brackets. +10% = squared errors 10% smaller than that simple rule.",
                VERDICT_GUIDE,
                "<b>5-model average</b>: the reference forecast used on the site."
                " <b>Gap</b> models forecast the surprise relative to the last-3 average."
                " <b>Constant</b>: the average of the whole family, a very weak benchmark except"
                " for the tail.",
            ],
        ),
        unsafe_allow_html=True,
    )
    comparison = ui.csv("new_sources_comparison.csv")
    extra = comparison[
        (comparison["family"] == family) & (comparison["model"] == "moyenne_5")
    ].set_index("target")
    rows = [
        [
            escape(ad.TARGET_LABELS[target]),
            en(row["toutes"] * 100, 1, sign=True, suffix="%"),
            en(row["sans_nouvelles_sources"] * 100, 1, sign=True, suffix="%"),
            ui.signed_number(row["gain"] * 100, 1)
            + f" [{en(row['gain_ci_low'] * 100, 1)}, {en(row['gain_ci_high'] * 100, 1)}]",
        ]
        for target, row in extra.iterrows()
    ]
    st.markdown(
        ui.panel(
            "Do the extra sources help?",
            ui.table(
                ["Measure", "With", "Without", "Difference (points)"], rows, right={1, 2, 3}
            ),  # fmt: skip
            "5-model average, 2014-2026. None of the differences is significantly positive:"
            " these sources do not improve the forecast, and slightly hurt some measures.",
            [
                "<b>Extra sources</b>: primary dealer positions (New York Fed), futures"
                " positioning (CFTC) and liquidity measures.",
                "<b>With / Without</b>: R² against the last-3 average with and without them."
                " <b>Difference</b>: with − without, and its 90% interval.",
            ],
        ),
        unsafe_allow_html=True,
    )

with tabs[1]:
    left, right = st.columns(2)
    target = left.selectbox("Measure", list(ad.TARGET_LABELS), format_func=ad.TARGET_LABELS.get)
    models = [m for m in ad.MODEL_LABELS if m in set(backtest["model"])]
    model = right.selectbox(
        "Model", models, format_func=ad.MODEL_LABELS.get,
        index=models.index("moyenne_5") if "moyenne_5" in models else 0,
    )  # fmt: skip
    guide = [
        "Running total of how much smaller the model's squared errors are than those of the"
        " last-3 average, as a share of the latter.",
        "Above 0, the model does better; the curve ends at the R² of the period. A steady"
        " climb means a regular edge; a jump means a few auctions explain the result.",
    ]
    if family == "bill":
        guide.append(
            "<b>Dashed lines</b>: debt-ceiling episodes, which explain part of the gain for bills."
        )
    with ui.box("gain", "Cumulative gain versus the last-3 average", guide):
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
        figure.update_layout(showlegend=False)
        figure.update_yaxes(ticksuffix="%")
        ui.chart(figure, height=420)

with tabs[2]:
    importance = ui.csv("feature_importance.csv")
    target = st.selectbox(
        "Measure", list(ad.TARGET_LABELS), format_func=ad.TARGET_LABELS.get, key="imp_target"
    )
    part = importance[(importance["family"] == family) & (importance["target"] == target)]
    if part.empty:
        st.markdown(
            ui.panel("What drives the forecast",
                     '<div class="tr-panel-note">No importance computed for this measure.</div>'),
            unsafe_allow_html=True,
        )  # fmt: skip
    else:
        ui.small(
            f"Model picked on 2014-2019: {escape(ad.MODEL_LABELS[part['model'].iloc[0]])},"
            " trained before 2022 and explained on 2022-2026. The two views can differ."
        )
        left, right = st.columns(2)
        with (
            left,
            ui.box(
                "permutation",
                "Permutation: what really helps",
                [
                    "Each variable is shuffled in turn: the longer the bar, the more the error"
                    " grows, so the more the variable really helps out of sample.",
                    "The thin line is the spread over repeated shuffles.",
                ],
            ),
        ):
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
            figure.update_xaxes(title="Increase in squared error when shuffled")
            ui.chart(figure, height=460)
        with (
            right,
            ui.box(
                "shap",
                "SHAP: what weighs in the formula",
                [
                    "Share of each variable in the model's contributions to its forecasts.",
                    "A variable can weigh a lot in the formula and still add little out of sample:"
                    " compare with the permutation view.",
                ],
            ),
        ):
            top = part.nlargest(12, "shap_share").iloc[::-1]
            figure = go.Figure(
                go.Bar(
                    x=top["shap_share"] * 100,
                    y=[ad.feature_label(name) for name in top["feature"]],
                    orientation="h",
                    marker_color=ui.MUTED,
                )
            )
            figure.update_xaxes(title="Share of total contributions", ticksuffix="%")
            ui.chart(figure, height=460)

with tabs[3]:
    ui.small(
        "Notes and bonds only. The strategy is short duration (it gains if the yield rises) from"
        " the previous close to the auction-day close when the bid-to-cover forecast is below the"
        " last-3 average, and long otherwise; 0.5 bp of cost per trade. Results in basis points"
        " (bp) of yield."
    )
    quintiles = ui.csv("quintiles.csv")
    full = quintiles[quintiles["period"] == "2009-2026"]
    y_range = [full["ci_low"].min() - 0.4, full["ci_high"].max() + 0.4]
    count = int(full.loc[full["surprise"] == "part_des_dealers", "n"].sum())
    guide = [
        "Auctions are split into 5 groups by surprise (actual − last-3 average): 1 = strongest"
        " demand, 5 = weakest.",
        "Each bar: average change in the yield of the auction's maturity from the previous close"
        " to the auction-day close, with its 90% interval.",
        f"Notes and bonds, 2009-2026 ({en(count, 0)} auctions).",
    ]
    columns = st.columns(2)
    for index, (column, (name, label)) in enumerate(
        zip(columns, ad.SURPRISE_LABELS.items(), strict=True)
    ):
        rows = full[full["surprise"] == name].sort_values("quintile")
        with column, ui.box(f"quintiles_{index}", label.split(" (")[0], guide):
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
            figure.update_xaxes(title="Surprise group (1 = strongest demand)")
            figure.update_yaxes(title="Yield change (bp)", range=y_range)
            ui.chart(figure, height=340)
    ui.small("The weakest auctions come with a rise in yields that day.")

    summary = ui.csv("strategy_summary.csv")
    rules = summary[summary["period"] == "2014-2026"].set_index("rule")

    def with_interval(value: float, low: float, high: float) -> str:
        if pd.isna(value):
            return "–"
        return f"{en(value, 2, sign=True)} [{en(low, 2)}, {en(high, 2)}]"

    rows = [
        [
            escape(ad.RULE_LABELS[rule]),
            en(row["trades"], 0),
            with_interval(row["mean_bp"], row["mean_ci_low"], row["mean_ci_high"]),
            en(row["mean_gross_bp"], 2, sign=True),
            with_interval(
                row["vs_vendeur_bp"], row["vs_vendeur_ci_low"], row["vs_vendeur_ci_high"]
            ),
            en(row["hit_rate"] * 100, 1, suffix="%"),
            en(row["total_bp"], 0, sign=True),
            en(row["sharpe"], 2, sign=True),
        ]
        for rule, row in rules.iterrows()
    ]
    loss = -rules.loc["strategie", "mean_bp"]
    perfect = rules.loc["information_parfaite", "mean_bp"]
    st.markdown(
        ui.panel(
            "Strategy results, 2014-2026",
            ui.table(
                [
                    "Rule",
                    "Trades",
                    "P&L per trade (bp)",
                    "Before cost",
                    "Versus always short",
                    "Hit rate",
                    "Total (bp)",
                    "Sharpe",
                ],
                rows,
                right={1, 2, 3, 4, 5, 6, 7},
                wrap={0},
            ),  # fmt: skip
            f"The strategy loses {en(loss, 1)} bp per trade after costs and does not beat always"
            " being short. Even knowing the auction result in advance would only earn"
            f" {en(perfect, 1, sign=True)} bp per trade after costs: the auction surprise explains"
            " only a small part of the day's yield move. Separately, the predictable part of"
            " demand has no measurable effect on yields.",
            [
                "<b>P&L per trade</b>: average gain after the 0.5 bp cost, with its 90% interval.",
                "<b>Versus always short</b>: average gain per trade over a rule that is always"
                " short, with its 90% interval.",
                "<b>Hit rate</b>: share of trades that gain after costs. <b>Sharpe</b>: average"
                " daily gain divided by its volatility, annualised.",
                "<b>Perfect foresight</b> knows the result in advance: an upper bound, impossible"
                " in practice.",
            ],
        ),
        unsafe_allow_html=True,
    )
    with ui.box(
        "pnl",
        "Cumulative P&L after costs (bp of yield)",
        ["Sum of the gains and losses of each rule, trade after trade, 2014-2026."],
    ):
        trades = ui.parquet("strategy_trades.parquet").sort_values("auction_date")
        figure = go.Figure()
        styles = {"strategie": (ui.SERIES[0], "solid"), "toujours_vendeur": (ui.SERIES[1], "solid"),
                  "information_parfaite": (ui.MUTED, "dash")}  # fmt: skip
        for rule, (color, dash) in styles.items():
            line = {"color": color, "dash": dash, "width": 2}
            figure.add_scatter(x=trades["auction_date"], y=trades[rule].cumsum(), mode="lines",
                               name=ad.RULE_LABELS[rule], line=line)  # fmt: skip
        ui.chart(figure, height=420)
