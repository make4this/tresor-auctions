"""Page « Methodology » : question, sources, variables, validation, résultats et limites.

Les chiffres cités sont lus dans les fichiers du backtest et de la stratégie.
"""

from __future__ import annotations

import streamlit as st

from tresor import app_data as ad
from tresor import app_ui as ui
from tresor.formats import en

r2 = ad.backtest_r2(ui.csv("backtest_metrics.csv")).set_index(["family", "target"])
summary = ui.csv("strategy_summary.csv").set_index(["period", "rule"])
coupons = en(r2.loc[("nominal_coupon", "btc"), "r2_vs_mean3"] * 100, 1, sign=True)
loss_all = en(-summary.loc[("2014-2026", "strategie"), "mean_bp"], 1)
loss_recent = en(-summary.loc[("2020-2026", "strategie"), "mean_bp"], 1)

ui.page_header(
    "Methodology",
    "How the forecasts are made and tested, what they show, and their limits.",
)


def section(title: str, items: list[str], intro: str = "") -> str:
    """Un panneau de texte : phrase d'introduction facultative et liste à puces."""
    lead = f"<p>{intro}</p>" if intro else ""
    bullets = "".join(f"<li>{item}</li>" for item in items)
    return ui.panel(title, f'<div class="tr-body">{lead}<ul>{bullets}</ul></div>')


SECTIONS = [
    section(
        "The question",
        [
            "the <b>bid-to-cover ratio</b> (amount bid / amount sold);",
            "the <b>dealer share</b>: what the primary dealers, banks required to bid, end up"
            " buying;",
            "the <b>indirect share</b>: bidders such as foreign central banks and funds;",
            "the <b>tail</b>: the auction yield minus the market yield of the day before.",
        ],
        "Before each US Treasury auction, how strong will demand be? Four measures are"
        " forecast, separately for notes and bonds (2 to 30 years) and bills (4 to 52 weeks):",
    ),
    section(
        "The data, all public and free",
        [
            "<b>US Treasury</b>: auction results since 2000 (Fiscal Data, TreasuryDirect), the"
            " auction calendar, the yield curve, bill rates, public debt and the Treasury's cash"
            " balance;",
            "<b>Cboe</b>: the VIX volatility index;",
            "<b>FRED</b> (St. Louis Fed): dollar index, Fed balance sheet, reverse repo, Baa"
            " credit spread, inflation expectations;",
            "<b>New York Fed</b>: primary dealer positions; <b>CFTC</b>: positioning in Treasury"
            " futures.",
        ],
    ),
    section(
        "Only what was known before the auction",
        [
            "Every value carries the date it became <b>public</b>, not the date it refers to.",
            "For an auction on day <i>t</i>, a variable only uses values published <b>strictly"
            " before</b> <i>t</i>; a blocking check rejects anything else.",
            "Where the archive of first releases exists (FRED's ALFRED), the real first-release"
            " date is used.",
        ],
    ),
    section(
        "How the models are tested",
        [
            "<b>Walk-forward</b>: every quarter since 2014, models are trained on all past"
            " auctions (since 2009) and tested on the next quarter.",
            "<b>Benchmark</b>: the average of the last 3 auctions of the same maturity.",
            "<b>Reference forecast</b>: the average of 5 models (Ridge, Lasso, ElasticNet,"
            " Random Forest, LightGBM), chosen on principle rather than on results.",
            "<b>Pick and judge separately</b>: the best single model is picked on 2014-2019 and"
            " judged on 2020-2026, with confidence intervals.",
        ],
    ),
    section(
        "Results",
        [
            "The <b>bid-to-cover of notes and bonds</b> is the most reliable result:"
            f" {coupons}% R² against the last-3 average over 2020-2026.",
            "The <b>extra sources</b> (dealers, CFTC, liquidity) improve no measure, and"
            " slightly hurt some.",
            "The <b>dealer share</b> of notes and bonds is poorly forecast since 2020; the"
            " <b>tail</b> mostly reflects the market move of the day.",
            f"A <b>trading strategy</b> based on the signal loses {loss_all} bp per trade after"
            f" costs over 2014-2026 ({loss_recent} bp over 2020-2026) and does not beat always"
            " being short.",
        ],
    ),
    section(
        "Limits",
        [
            "The Treasury announces auctions only a few days ahead, sometimes before their size"
            " (the size is then estimated).",
            "The tail is approximated with the previous close: the true benchmark, the 1 pm"
            " when-issued yield, is paid data.",
            "The 2015, 2021 and 2023 debt-ceiling episodes inflate part of the results for bills.",
            "The 0.5 bp cost per trade is an assumption.",
        ],
    ),
]

for start in range(0, len(SECTIONS), 2):
    left, right = st.columns(2, gap="medium")
    left.markdown(SECTIONS[start], unsafe_allow_html=True)
    if start + 1 < len(SECTIONS):
        right.markdown(SECTIONS[start + 1], unsafe_allow_html=True)

st.markdown(
    section(
        "Glossary",
        [
            "<b>bp</b>: basis point, 0.01 percentage point of yield.",
            "<b>R² against the last-3 average</b>: 1 − (model squared error / squared error of"
            " the last-3 average); above 0, the model does better.",
            "<b>Duration</b>: sensitivity of a bond to yields; <i>short duration</i> gains when"
            " yields rise.",
            "<b>Permutation / SHAP</b>: what a variable adds out of sample / how much it weighs"
            " in the model.",
        ],
    ),
    unsafe_allow_html=True,
)
